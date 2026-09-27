import base64
import io
import json
import os
import time
from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool
from tritonclient.utils import InferenceServerException

TRITON_URL = os.getenv("TRITON_URL", "triton:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "image_classifier")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
CLASSES_PATH = os.getenv("CLASSES_PATH", "/models/image_classifier/class_names.json")

app = FastAPI(
    title="Image Classification Gateway",
    description="FastAPI gateway for NVIDIA Triton image classification model (animals)",
    version="1.0.0",
)

triton_client: grpcclient.InferenceServerClient | None = None
input_name: str | None = None
output_name: str | None = None
class_names: list[str] = []


def load_classes() -> list[str]:
    if not os.path.exists(CLASSES_PATH):
        raise RuntimeError(f"Classes file not found: {CLASSES_PATH}")
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    ВАЖНО: модель (EfficientNetB0) содержит встроенные слои Rescaling + Normalization
    (ImageNet mean/std), поэтому на вход подаются СЫРЫЕ пиксели [0, 255], float32,
    БЕЗ деления на 255 и без ручной нормализации — это уже часть ONNX-графа.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(image, dtype=np.float32)  # [0, 255], без /255.0


def decode_base64_image(value: str) -> bytes:
    if "," in value and value.strip().lower().startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def infer(batch: np.ndarray) -> np.ndarray:
    if triton_client is None or input_name is None or output_name is None:
        raise RuntimeError("Triton client is not initialized")

    infer_input = grpcclient.InferInput(input_name, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch.astype(np.float32))
    infer_output = grpcclient.InferRequestedOutput(output_name)

    response = triton_client.infer(
        model_name=MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output],
    )
    result = response.as_numpy(output_name)
    if result is None:
        raise RuntimeError(f"Triton returned empty output: {output_name}")
    return result


@app.on_event("startup")
def startup() -> None:
    global triton_client, input_name, output_name, class_names

    class_names = load_classes()
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    last_error: Exception | None = None
    for _ in range(30):
        try:
            if triton_client.is_server_live() and triton_client.is_model_ready(MODEL_NAME):
                metadata = triton_client.get_model_metadata(MODEL_NAME)
                input_name = metadata.inputs[0].name
                output_name = metadata.outputs[0].name
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2)

    raise RuntimeError(f"Triton model is not ready: {last_error}")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Image Classification Gateway",
        "model": MODEL_NAME,
        "endpoints": ["GET /health", "GET /classes", "POST /predict"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        live = triton_client.is_server_live() if triton_client else False
        ready = triton_client.is_model_ready(MODEL_NAME) if triton_client else False
        return {
            "status": "healthy" if live and ready else "unhealthy",
            "triton_live": live,
            "model_ready": ready,
            "model_name": MODEL_NAME,
            "input_name": input_name,
            "output_name": output_name,
            "classes_loaded": len(class_names),
        }
    except InferenceServerException as exc:
        return {"status": "unhealthy", "error": str(exc)}


@app.get("/classes")
def classes() -> dict[str, Any]:
    return {"classes": class_names}


@app.post("/predict")
async def predict(
    file: UploadFile | None = File(None, description="Файл изображения (jpg/png)"),
    image: str | None = Form(None, description="Изображение в base64 — альтернатива file"),
) -> dict[str, Any]:
    """
    Явно объявленные параметры (а не «сырой» Request) нужны, чтобы Swagger UI сам
    нарисовал поле загрузки файла в /docs. Прими файл ЛИБО base64-строку в поле image.
    """
    if file is not None:
        image_bytes = await file.read()
    elif image is not None:
        image_bytes = decode_base64_image(image)
    else:
        raise HTTPException(status_code=400, detail="Provide 'file' or 'image'")

    try:
        array = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    batch = np.expand_dims(array, axis=0)

    try:
        start = time.perf_counter()
        # ВАЖНО: triton_client.infer() — синхронный (блокирующий) gRPC-вызов.
        # Без run_in_threadpool он выполнялся бы прямо в event loop и фактически
        # сериализовал бы параллельные запросы, не давая Dynamic Batching
        # на стороне Triton собрать их в один батч — тест показывал бы
        # искажённые (заниженные) throughput и латентность.
        scores = await run_in_threadpool(infer, batch)
        inference_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    row = scores[0]
    class_id = int(np.argmax(row))

    return {
        "model": MODEL_NAME,
        "inference_ms": round(inference_ms, 3),
        "predicted_class": class_names[class_id],
        "confidence": float(row[class_id]),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(class_names, row)
        },
    }
