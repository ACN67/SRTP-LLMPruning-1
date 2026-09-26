ARG BASE_IMAGE=docker.m.daocloud.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG SRTP_PROJECT_GIT_COMMIT=unknown
ARG SRTP_SOURCE_DIRTY=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    HF_DATASETS_CACHE=/data/cache/huggingface/datasets \
    SRTP_PROJECT_GIT_COMMIT=${SRTP_PROJECT_GIT_COMMIT} \
    SRTP_SOURCE_DIRTY=${SRTP_SOURCE_DIRTY}

LABEL org.opencontainers.image.revision="${SRTP_PROJECT_GIT_COMMIT}" \
      io.srtp.source.dirty="${SRTP_SOURCE_DIRTY}"

WORKDIR /workspace

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt && \
    python -c "import torch; assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda"

COPY . .
RUN mkdir -p /data/models /data/cache /data/datasets /data/checkpoints /data/results
RUN python -m unittest discover -s tests -v && \
    python -m compileall -q src scripts tests && \
    python scripts/setup/preflight.py --software-only

CMD ["python", "scripts/setup/preflight.py", "--software-only"]
