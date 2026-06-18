FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    # FIX: lần build image này chưa có model ProtonX trong cache, nên cho
    # phép đúng 1 lần "warm-up" online khi service khởi động lần đầu để
    # download + cache model (paddle_ocr_processor.py sẽ tự khôi phục lại
    # HF_HUB_OFFLINE=1 ngay sau warm-up, để các request sau không còn gọi
    # mạng). Nếu bạn bake sẵn model vào image (xem ghi chú dưới), có thể
    # override lại thành "false" khi `docker run -e PROTONX_ALLOW_ONLINE_WARMUP=false`.
    PROTONX_ALLOW_ONLINE_WARMUP=true

# ===== SWITCH APT TO HTTPS =====
RUN printf 'deb https://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse\n\
deb https://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse\n\
deb https://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse\n\
deb https://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse\n' \
> /etc/apt/sources.list \
&& apt-get -o Acquire::https::Verify-Peer=false update \
&& apt-get install -y --no-install-recommends apt-transport-https ca-certificates \
&& rm -rf /var/lib/apt/lists/*

# ===== SYSTEM DEPENDENCIES =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    python3-pip \
    tesseract-ocr tesseract-ocr-vie \
    poppler-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    wget curl git dos2unix \
    fonts-liberation fonts-dejavu \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# ===== PYTHON SETUP =====
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip dùng module pip có sẵn (không cần bootstrap)
RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && update-alternatives --install /usr/bin/pip pip $(python3.10 -m pip show pip 2>/dev/null | grep Location | awk '{print $2}' | sed 's|/lib/.*||')/bin/pip3.10 1 2>/dev/null || true

WORKDIR /app
RUN mkdir -p /app/hf_cache

COPY requirements.txt /app/requirements.txt

# FIX: requirements.txt giờ chứa "--find-links" (cho paddlepaddle-gpu) và
# "--extra-index-url" (cho torch+cu121) ở nhiều dòng khác nhau — pip hỗ trợ
# việc này trực tiếp trong file requirements (PEP 508 / pip global options
# theo dòng), nên không cần thay đổi gì thêm ở lệnh install, chỉ cần đảm bảo
# dos2unix/sed dọn file đúng như trước.
RUN dos2unix /app/requirements.txt || true && \
    sed -i '1s/^\xEF\xBB\xBF//' /app/requirements.txt && \
    python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/uploads /app/models /app/hf_cache

EXPOSE 8000 8001 8501

CMD ["bash", "-lc", "python3.10 -c \"import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); import paddle; print('paddle:', paddle.__version__); print('paddle gpu:', paddle.device.is_compiled_with_cuda())\""]