import io
import os

import matplotlib.pyplot as plt
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

API_URL = os.getenv("API_URL", "http://api:8080/predict")  # http://api:8080 - имя сервиса в docker-compose
CANVAS_SIZE = 280

st.set_page_config(page_title="Классификация животных (Triton)", page_icon="🐾", layout="centered")
st.title("🐾 Классификация изображений животных")
st.caption("Backend: Triton Inference Server + FastAPI Gateway · EfficientNetB0")

tab_upload, tab_draw = st.tabs(["📤 Загрузить изображение", "✏️ Нарисовать"])

raw_image = None

with tab_upload:
    uploaded_file = st.file_uploader("Выбери изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        st.image(raw_image, caption="Загруженное изображение", width=280)

with tab_draw:
    st.write("Нарисуй что-нибудь на холсте (белым по чёрному фону):")
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        return_image_data=True,
        key="canvas",
    )
    if canvas_result.image_data is not None and canvas_result.image_data[:, :, :3].sum() > 0:
        raw_image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")

st.divider()


def to_png_bytes(img: Image.Image) -> bytes:
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = background
    else:
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if raw_image is not None:
    if st.button("Классифицировать", type="primary"):
        with st.spinner("Отправляю на Triton через FastAPI Gateway..."):
            image_bytes = to_png_bytes(raw_image)
            try:
                response = requests.post(
                    API_URL,
                    files={"file": ("image.png", image_bytes, "image/png")},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка запроса к API: {e}")
                result = None

        if result is not None:
            inference_ms = result.get("inference_ms")

            st.success(
                f"Предсказанный класс: **{result['predicted_class']}** "
                f"(уверенность: {result['confidence']:.1%})"
            )

            if inference_ms is not None:
                st.metric("Время инференса (Triton)", f"{inference_ms:.1f} мс")

            probs = result["probabilities"]
            sorted_probs = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))

            # Рисуем через matplotlib, а не st.bar_chart/st.dataframe — те под капотом
            # используют pyarrow, который на некоторых Anaconda-окружениях на macOS
            # падает из-за конфликта версий libprotobuf/liborc.
            fig, ax = plt.subplots(figsize=(6, 3))
            classes = list(sorted_probs.keys())
            values = list(sorted_probs.values())
            ax.barh(classes, values, color="#4C78A8")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Вероятность")
            ax.invert_yaxis()
            for i, v in enumerate(values):
                ax.text(min(v + 0.02, 0.95), i, f"{v:.1%}", va="center")
            fig.tight_layout()
            st.pyplot(fig)

            for cls, p in sorted_probs.items():
                st.write(f"**{cls}**: {p:.1%}")
else:
    st.info("Загрузи изображение или нарисуй его на холсте, чтобы начать.")
