# Triton Image Classification Stack

Полнофункциональный проект разворачивания модели классификации изображений через NVIDIA Triton Inference Server. Решение включает FastAPI Gateway, Swagger UI, Streamlit-интерфейс, Prometheus и Grafana-дэшборд.

Модель классифицирует изображения животных по пяти классам:

- `butterfly`
- `cow`
- `elephant`
- `sheep`
- `squirrel`

## Преимущества архитектуры

### NVIDIA Triton Inference Server

- Запускает ONNX-модель (EfficientNetB0) через backend `onnxruntime`.
- Поддерживает dynamic batching для повышения throughput.
- Экспортирует метрики Prometheus на порту `8002`, включая Summary-метрики латентности (`--metrics-config summary_latencies=true`).
- Хранит модель в стандартной структуре `model_repository/<model_name>/<version>/model.onnx`.

### FastAPI Gateway

- Принимает изображения через `multipart/form-data` (`file`) или form-поле base64 (`image`).
- Препроцессинг: RGB, resize `224x224`, сырые пиксели `[0, 255]` — нормализация (ImageNet mean/std) внутри самой ONNX-модели.
- Вызывает Triton по gRPC через threadpool (`run_in_threadpool`), чтобы блокирующий вызов не сериализовал параллельные запросы и не мешал Dynamic Batching.
- Возвращает предсказанный класс, confidence, probabilities и время инференса.
- Предоставляет Swagger UI для ручной проверки `/predict` с явными полями `file`/`image`.

### Streamlit UI

- Позволяет загрузить изображение или нарисовать его на холсте.
- Отображает предсказанный класс, уверенность, время инференса и график вероятностей по классам (через `matplotlib`, без `pyarrow`).
- Запускается одной командой вместе с остальным стеком через Docker Compose.

### Prometheus + Grafana

- Prometheus собирает метрики Triton.
- Grafana автоматически получает datasource и dashboard через provisioning.
- Дэшборд показывает RPS, latency (p50/p95/p99), queue time, pending requests и total requests.

## Стек

| Компонент | Назначение |
|---|---|
| NVIDIA Triton | Инференс ONNX-модели |
| FastAPI | REST API Gateway |
| Streamlit | Пользовательский веб-интерфейс |
| Prometheus | Сбор метрик |
| Grafana | Визуализация метрик |
| Docker Compose | Запуск всего стека |

## Структура проекта

```text
triton-classification/
├── docker-compose.yml
├── README.md
├── test_client.py
├── benchmark_dynamic_batching.py
│
├── model_repository/
│   └── image_classifier/
│       ├── config.pbtxt
│       ├── class_names.json
│       └── 1/
│           └── model.onnx
│
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
│
├── streamlit_app/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
│
├── prometheus/
│   └── prometheus.yml
│
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── datasources.yml
        └── dashboards/
            ├── dashboards.yml
            └── triton.json
```

Папка `results/` не хранится в репозитории и создаётся локально при запуске `benchmark_dynamic_batching.py`.

## Быстрый старт

```bash
git clone <ссылка-на-ваш-репозиторий>
cd triton-classification
docker compose up -d --build
```

Если используется старая команда Compose:

```bash
docker-compose up -d --build
```

## Проверка запуска

```bash
docker compose ps
curl http://localhost:8080/health
curl http://localhost:8000/v2/models/image_classifier
```

Ожидаемый ответ `/health`:

```json
{
  "status": "healthy",
  "triton_live": true,
  "model_ready": true,
  "model_name": "image_classifier",
  "input_name": "input_image",
  "output_name": "dense_1",
  "classes_loaded": 5
}
```

Ожидаемые метаданные модели:

```json
{
  "name": "image_classifier",
  "versions": ["1"],
  "platform": "onnxruntime_onnx",
  "inputs": [{"name": "input_image", "datatype": "FP32", "shape": [-1, 224, 224, 3]}],
  "outputs": [{"name": "dense_1", "datatype": "FP32", "shape": [-1, 5]}]
}
```

## Сервисы

| Сервис | URL | Описание |
|---|---|---|
| FastAPI | http://localhost:8080 | REST API |
| Swagger UI | http://localhost:8080/docs | Документация и ручной тест `/predict` |
| Streamlit | http://localhost:8501 | Веб-интерфейс классификации |
| Triton HTTP | http://localhost:8000 | Triton HTTP API |
| Triton gRPC | localhost:8001 | Triton gRPC API |
| Triton Metrics | http://localhost:8002/metrics | Метрики для Prometheus |
| Prometheus | http://localhost:9090 | Web UI Prometheus |
| Grafana | http://localhost:3000 | Дэшборды, `admin/admin` |

## API Endpoints

| Метод | Endpoint | Описание |
|---|---|---|
| `GET` | `/` | Информация об API |
| `GET` | `/health` | Проверка состояния Triton и модели |
| `GET` | `/classes` | Список классов |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/predict` | Классификация изображения |

### Пример запроса через файл

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@path/to/image.jpg;type=image/jpeg"
```

### Пример через base64 (form-поле `image`, альтернатива `file`)

Оба поля видны и заполняемы прямо в Swagger UI (`/docs` → `POST /predict` → Try it out).

### Пример ответа

```json
{
  "model": "image_classifier",
  "inference_ms": 170.9,
  "predicted_class": "squirrel",
  "confidence": 1.0,
  "probabilities": {
    "butterfly": 0.0,
    "cow": 0.0,
    "elephant": 0.0,
    "sheep": 0.0,
    "squirrel": 1.0
  }
}
```

## Конфигурация Triton

Файл: `model_repository/image_classifier/config.pbtxt`

```text
name: "image_classifier"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input_image"
    data_type: TYPE_FP32
    dims: [224, 224, 3]
  }
]

output [
  {
    name: "dense_1"
    data_type: TYPE_FP32
    dims: [5]
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100000
}

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

## Streamlit

Streamlit-интерфейс поднимается вместе с основным стеком:

```bash
docker compose up -d --build
```

Открыть интерфейс:

```text
http://localhost:8501
```

Интерфейс позволяет загрузить изображение животного (или нарисовать его на холсте), выполнить классификацию через FastAPI/Triton и увидеть класс, уверенность, время инференса и вероятности по классам.

## Мониторинг

Prometheus собирает метрики Triton с адреса:

```text
triton:8002/metrics
```

Grafana автоматически загружает:

- datasource: `grafana/provisioning/datasources/datasources.yml`
- dashboard: `grafana/provisioning/dashboards/triton.json`


Основные панели:

| Панель | Метрика |
|---|---|
| Requests per Second | `rate(nv_inference_request_success{model="image_classifier"}[1m])` / `rate(nv_inference_request_failure{model="image_classifier"}[1m])` |
| Inference Latency (p50/p95/p99) | `nv_inference_request_summary_us{model="image_classifier", quantile="0.5"\|"0.95"\|"0.99"}` |
| Average Queue Time | `rate(nv_inference_queue_duration_us{model="image_classifier"}[1m]) / rate(nv_inference_count{model="image_classifier"}[1m])` |
| Pending Requests | `nv_inference_pending_request_count{model="image_classifier"}` |
| Total Requests | `nv_inference_request_success` и `nv_inference_request_failure` |


## Нагрузочное тестирование

Скрипт:

```bash
python3 -u benchmark_dynamic_batching.py \
  --image "path/to/animal.jpg" \
  --requests 1000 \
  --concurrency 50 \
  --warmup 20 \
  --mode all \
  --restore
```

Папка `results/` создаётся локально при запуске benchmark-скрипта и не хранится в репозитории.

Перед прогоном в `main.py` уже устранён источник искажений: блокирующий вызов Triton (`triton_client.infer(...)`) вынесен в threadpool через `run_in_threadpool` — без этого `async`-эндпоинт частично сериализовал бы параллельные запросы и не позволял Dynamic Batching реально собирать батч.

Результаты:

| Конфигурация | preferred_batch_size | max_queue_delay_microseconds | Avg Latency (ms) | P95 (ms) | P99 (ms) | Throughput (RPS) | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| No batching | - | - | 1556.33 | 2202.02 | 2627.39 | 31.34 | 1000/1000 |
| Small batch | [2, 4] | 50000 | 750.39 | 1090.17 | 1275.53 | 65.67 | 1000/1000 |
| Medium batch | [4, 8, 16] | 100000 | 659.38 | 921.12 | 1043.99 | 74.68 | 1000/1000 |
| Large batch | [8, 16, 32] | 200000 | 684.93 | 1039.97 | 1150.74 | 71.78 | 1000/1000 |

Вывод: dynamic batching дал прирост throughput в 2.1–2.4 раза и такое же снижение средней latency. Наилучший баланс между throughput и latency показала конфигурация `[4, 8, 16]` с `max_queue_delay_microseconds=100000` — она одновременно лучшая и по throughput, и по latency. Конфигурация `[8, 16, 32]` оказалась немного хуже средней: слишком большая задержка очереди (200 мс) заставляет Triton дольше ждать заполнения крупного батча, чем реально успевает прийти запросов при параллельности клиента 50.

Абсолютные значения латентности выше типичных ориентиров из-за среды выполнения: инференс идёт на CPU (`KIND_CPU`) внутри Docker на Apple Silicon Mac, где образ Triton (`linux/amd64`) запускается через эмуляцию, а не нативно на Linux с GPU.

## Поддержка GPU

В текущей версии проект настроен на CPU:

```text
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
```

Для запуска на NVIDIA GPU нужно установить NVIDIA Container Toolkit и заменить `instance_group` на `KIND_GPU`, а также добавить GPU-reservation в `docker-compose.yml`.

## Требования

- Docker Desktop / Docker Engine
- Docker Compose
- 8 GB RAM желательно для комфортного запуска Triton, Grafana, Prometheus, FastAPI и Streamlit
- macOS/Linux/Windows с Docker

## Остановка

```bash
docker compose stop
```

Полное удаление контейнеров:

```bash
docker compose down
```
