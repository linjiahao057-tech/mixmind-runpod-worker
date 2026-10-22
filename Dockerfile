FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# 使用阿里云镜像源加速，并且剔除 torch/torchaudio (因为基础镜像已有)
RUN pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir --ignore-installed -i https://mirrors.aliyun.com/pypi/simple/ \
      runpod \
      transformers \
      sentencepiece \
      librosa \
      soundfile && \
    pip install --no-cache-dir --no-deps -i https://mirrors.aliyun.com/pypi/simple/ \
      "qwen-asr"

COPY rp_handler.py /app/rp_handler.py

CMD ["python3", "-u", "rp_handler.py"]
