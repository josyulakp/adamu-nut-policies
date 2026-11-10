# Hugging Face Space (Docker SDK) for the interactive eval page: the server and both workers, CPU-only by default.
# Built from a folder staged by space/make_space.sh. The trained models are downloaded from the model repo at build.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git libosmesa6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.1 "huggingface_hub>=1.0"

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user/app

# TORCH=cpu (free CPU Space) or cu121 for a GPU Space (Touch in the Wild / DexScrew env; the main env then uses cu132)
ARG TORCH=cpu
COPY --chown=user setup/ setup/
RUN set -e; for f in main tw; do \
      grep -vE '^(nvidia-|cuda-|triton)' setup/requirements-$f.txt | sed -E 's/\+cu[0-9]+//' > /tmp/req-$f.txt; done; \
    if [ "$TORCH" = cpu ]; then IM=cpu; IT=cpu; else IM=cu132; IT=cu121; fi; \
    uv venv -q --python /usr/local/bin/python3.11 .venv && \
    uv pip install --no-cache -q --python .venv/bin/python -r /tmp/req-main.txt \
      --extra-index-url https://download.pytorch.org/whl/$IM --index-strategy unsafe-best-match && \
    uv venv -q --python /usr/local/bin/python3.11 .venv-tw && \
    uv pip install --no-cache -q --python .venv-tw/bin/python -r /tmp/req-tw.txt \
      --extra-index-url https://download.pytorch.org/whl/$IT --index-strategy unsafe-best-match

# upstream code, at the commits this code was developed against (same as setup/setup.sh)
RUN git clone -q https://github.com/XinyueZhuXY/touch_in_the_wild ext/touch_in_the_wild && \
    git -C ext/touch_in_the_wild checkout -q 2a946261d34bee5ac363c5dc696810273a0b20d8 && \
    git clone -q https://github.com/x-robotics-lab/dexscrew ext/dexscrew && \
    git -C ext/dexscrew checkout -q 3bde4e3a4d973743921c75719ca88167de144e83

# trained models + evaluation results (for the success rates the page shows)
ARG MODEL_REPO=josyula/adamu-nut-policies
RUN hf download $MODEL_REPO --include "models/*" --include "results/*" --local-dir /home/user/data && rm -rf /home/user/data/.cache

COPY --chown=user . .

ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ADAMU_DATA=/home/user/data HOST=0.0.0.0 PORT=7860 \
    OMP_NUM_THREADS=2 TORCH_THREADS=2 MAX_MODELS=3 PYTHONUNBUFFERED=1 \
    RENDER_MAX_FACES=2000 RENDER_SHADOW=1024
EXPOSE 7860
CMD ["bash", "webapp/run.sh"]
