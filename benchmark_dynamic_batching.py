"""
Нагрузочное тестирование API с автоматическим перебором конфигураций Dynamic Batching.

Для каждой конфигурации скрипт:
1. Переписывает model_repository/image_classifier/config.pbtxt
2. Перезапускает контейнер Triton и ждёт готовности модели
3. Прогревает API (--warmup запросов), затем шлёт --requests запросов
   с параллельностью --concurrency через aiohttp
4. Считает avg/p95/p99 латентность и throughput (RPS)
5. Сохраняет результаты в results/results.csv и results/comparison.png

Запуск (из корня triton-classification/):

    python3 -u benchmark_dynamic_batching.py \
        --image path/to/test_image.jpg \
        --requests 1000 \
        --concurrency 50 \
        --warmup 20 \
        --mode all
"""

import argparse
import asyncio
import subprocess
import time
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "model_repository" / "image_classifier" / "config.pbtxt"
RESULTS_DIR = PROJECT_DIR / "results"
API_URL = "http://localhost:8080/predict"
TRITON_HEALTH_URL = "http://localhost:8000/v2/health/ready"

# Базовая часть config.pbtxt (без блока dynamic_batching) — под модель животных 224x224x3, 5 классов
BASE_CONFIG = """name: "image_classifier"
backend: "onnxruntime"
max_batch_size: 32

input [
  {{
    name: "input_image"
    data_type: TYPE_FP32
    dims: [224, 224, 3]
  }}
]

output [
  {{
    name: "dense_1"
    data_type: TYPE_FP32
    dims: [5]
  }}
]

{dynamic_batching_block}
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
"""

CONFIGURATIONS = [
    {
        "name": "Без батчинга",
        "preferred_batch_size": None,
        "max_queue_delay_microseconds": None,
    },
    {
        "name": "Малый батч",
        "preferred_batch_size": [2, 4],
        "max_queue_delay_microseconds": 50000,
    },
    {
        "name": "Средний батч",
        "preferred_batch_size": [4, 8, 16],
        "max_queue_delay_microseconds": 100000,
    },
    {
        "name": "Большой батч",
        "preferred_batch_size": [8, 16, 32],
        "max_queue_delay_microseconds": 200000,
    },
]


def render_config(preferred_batch_size, max_queue_delay_microseconds) -> str:
    if preferred_batch_size is None:
        block = ""
    else:
        batch_str = ", ".join(str(v) for v in preferred_batch_size)
        block = (
            "dynamic_batching {\n"
            f"  preferred_batch_size: [{batch_str}]\n"
            f"  max_queue_delay_microseconds: {max_queue_delay_microseconds}\n"
            "}\n\n"
        )
    return BASE_CONFIG.format(dynamic_batching_block=block)


def apply_config(preferred_batch_size, max_queue_delay_microseconds) -> None:
    CONFIG_PATH.write_text(
        render_config(preferred_batch_size, max_queue_delay_microseconds),
        encoding="utf-8",
    )


def restart_triton(timeout_s: int = 90) -> None:
    subprocess.run(["docker", "compose", "restart", "triton"], cwd=PROJECT_DIR, check=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(TRITON_HEALTH_URL, timeout=3).status_code == 200:
                time.sleep(3)  # небольшой запас, чтобы модель точно прогрузилась
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("Triton не поднялся вовремя после restart")


async def send_request(session: aiohttp.ClientSession, image_bytes: bytes) -> tuple[bool, float]:
    start = time.perf_counter()
    try:
        form = aiohttp.FormData()
        form.add_field("file", image_bytes, filename="test.jpg", content_type="image/jpeg")
        async with session.post(API_URL, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            await resp.read()
            ok = resp.status == 200
    except Exception:
        ok = False
    elapsed_ms = (time.perf_counter() - start) * 1000
    return ok, elapsed_ms


async def run_load_test(image_bytes: bytes, total_requests: int, concurrency: int, warmup: int) -> dict:
    async with aiohttp.ClientSession() as session:
        # Прогрев
        for _ in range(warmup):
            await send_request(session, image_bytes)

        latencies: list[float] = []
        successes = 0
        failures = 0
        semaphore = asyncio.Semaphore(concurrency)

        async def worker():
            nonlocal successes, failures
            async with semaphore:
                ok, elapsed_ms = await send_request(session, image_bytes)
                latencies.append(elapsed_ms)
                if ok:
                    successes += 1
                else:
                    failures += 1

        start = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(total_requests)])
        total_time_s = time.perf_counter() - start

    latencies_arr = np.array(latencies)
    return {
        "avg_latency_ms": float(np.mean(latencies_arr)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
        "throughput_rps": total_requests / total_time_s,
        "successes": successes,
        "failures": failures,
    }


def plot_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(df["configuration"], df["throughput_rps"], color="#4C78A8")
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("RPS")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(df["configuration"], df["avg_latency_ms"], marker="o", label="Avg")
    axes[1].plot(df["configuration"], df["p95_ms"], marker="o", label="P95")
    axes[1].plot(df["configuration"], df["p99_ms"], marker="o", label="P99")
    axes[1].set_title("Latency")
    axes[1].set_ylabel("ms")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "comparison.png", dpi=150, bbox_inches="tight")
    print(f"График сохранён: {RESULTS_DIR / 'comparison.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Путь к тестовому изображению")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mode", choices=["all", "current"], default="all",
                         help="'all' — прогнать все 4 конфигурации; 'current' — только текущий config.pbtxt")
    parser.add_argument("--restore", action="store_true",
                         help="Вернуть config.pbtxt к состоянию 'Средний батч' после завершения теста")
    args = parser.parse_args()

    image_bytes = Path(args.image).read_bytes()
    RESULTS_DIR.mkdir(exist_ok=True)

    configs_to_run = CONFIGURATIONS if args.mode == "all" else [None]
    rows = []

    for cfg in configs_to_run:
        if cfg is not None:
            print(f"\n=== Конфигурация: {cfg['name']} ===")
            apply_config(cfg["preferred_batch_size"], cfg["max_queue_delay_microseconds"])
            print("Перезапуск Triton...")
            restart_triton()

        result = asyncio.run(
            run_load_test(image_bytes, args.requests, args.concurrency, args.warmup)
        )
        print(result)

        rows.append({
            "configuration": cfg["name"] if cfg else "current",
            "preferred_batch_size": cfg["preferred_batch_size"] if cfg else None,
            "max_queue_delay_microseconds": cfg["max_queue_delay_microseconds"] if cfg else None,
            **result,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    print(f"\nРезультаты сохранены: {RESULTS_DIR / 'results.csv'}")
    print(df)

    if args.mode == "all":
        plot_comparison(df)

    if args.restore:
        medium = CONFIGURATIONS[2]
        apply_config(medium["preferred_batch_size"], medium["max_queue_delay_microseconds"])
        restart_triton()
        print("config.pbtxt восстановлен к 'Средний батч'")


if __name__ == "__main__":
    main()
