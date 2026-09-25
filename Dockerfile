ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    HF_DATASETS_CACHE=/data/cache/huggingface/datasets

WORKDIR /workspace

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /data/models /data/cache /data/datasets /data/checkpoints /data/results

CMD ["python", "scripts/validate_model.py", "--help"]
