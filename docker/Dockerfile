# syntax=docker/dockerfile:1.7
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
ARG CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04@sha256:a5dc56795348ea8fe46305612afaf7f9882cb438b08a582d4e5bd762369d7922
FROM node:22-bookworm-slim@sha256:9f0dde62d10e7ff7a310ecaebd66225f6b46c17e6cb778449a77b05d3541bea2 AS node

FROM ${CUDA_IMAGE} AS system
ARG MINIFORGE_VERSION=26.9.4-1
ARG MINIFORGE_SHA256=571b410272cc434f28bc6c5c49ca8ffd4a4d1bcba9f57f1b6baccdb46446cfcc
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/envs/py312/bin:/opt/embodichain/media/bin:/opt/conda/bin:$PATH \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,display \
    EMBODICHAIN_ENV=py312 \
    EMBODICHAIN_BLENDER_PYTHON=/opt/conda/envs/blender311/bin/python \
    EMBODICHAIN_CONSTRAINTS_DIR=/opt/embodichain/locks \
    PYTHONNOUSERSITE=1
# Keep JIT headers/NVRTC used by Warp/cuRobo, plus the native graphics closure
# of DexSim and Blender. Driver libraries are supplied by Container Toolkit.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git gcc g++ make \
    cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 cuda-cccl-12-8 \
    libegl1 libgl1 libglx0 libopengl0 libglu1-mesa libglvnd0 libvulkan1 \
    libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 \
    libxxf86vm1 libsm6 libice6 libglib2.0-0 libgomp1 libxkbcommon0 \
    libxcb1 libxcb-cursor0 libxcb-xinerama0 libfontconfig1 libfreetype6 \
    libdbus-1-3 libnss3 libasound2 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY docker/locks/conda-*.txt /opt/embodichain/locks/
RUN (curl -fL --retry 3 \
    "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    -o /tmp/miniforge.sh \
    || curl -fL --retry 3 \
    "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh" \
    -o /tmp/miniforge.sh) \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum -c - \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
    && /opt/conda/bin/python -m pip install --no-cache-dir uv==0.12.5 \
    && conda create -y -n py312 --file /opt/embodichain/locks/conda-py312.txt \
    && conda create -y -p /opt/embodichain/media --file /opt/embodichain/locks/conda-media.txt \
    && conda clean -afy
# Media has no business Python packages: exposing its shared FFmpeg libraries
# never makes one business environment load the other's DexSim/libpython.
ENV LD_LIBRARY_PATH=/opt/embodichain/media/lib:$LD_LIBRARY_PATH
COPY docker/locks/py311.txt docker/locks/py312.txt docker/locks/blender311.txt /opt/embodichain/locks/
COPY docker/install-profile.sh docker/entrypoint.sh docker/verify.py docker/repair-bpy-wheel.py /opt/embodichain/
COPY docker/profile.sh /etc/profile.d/embodichain.sh

FROM system AS dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    bash /opt/embodichain/install-profile.sh py312

FROM dependencies AS wheel-builder
WORKDIR /src
COPY . /src
RUN uv build --wheel --out-dir /wheels

FROM dependencies AS public
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source=https://github.com/DexForce/EmbodiChain \
    org.opencontainers.image.revision=${SOURCE_REVISION} \
    org.opencontainers.image.title="EmbodiChain Python 3.12 runtime"
COPY --from=wheel-builder /wheels/ /tmp/wheels/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/py312/bin/python --no-deps /tmp/wheels/*.whl \
    && conda create -y -n blender311 --file /opt/embodichain/locks/conda-py311.txt \
    && bash /opt/embodichain/install-profile.sh blender311 \
    && conda clean -afy && rm -rf /tmp/wheels \
    && uv pip check --python /opt/conda/envs/py312/bin/python \
    && mkdir -p /workspace \
    && printf '\nsource /etc/profile.d/embodichain.sh\n' >> /root/.bashrc
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "/opt/embodichain/entrypoint.sh"]
CMD ["/bin/bash"]

FROM public AS internal
LABEL org.opencontainers.image.title="EmbodiChain Python 3.11/3.12 CI runtime"
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
COPY --from=wheel-builder /wheels/ /tmp/wheels/
RUN --mount=type=cache,target=/root/.cache/uv \
    ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && conda create -y -n py311 --file /opt/embodichain/locks/conda-py311.txt \
    && bash /opt/embodichain/install-profile.sh py311 \
    && uv pip install --python /opt/conda/envs/py311/bin/python --no-deps /tmp/wheels/*.whl \
    && for profile in py311 py312; do \
         uv pip install --python "/opt/conda/envs/${profile}/bin/python" \
           -c "/opt/embodichain/locks/${profile}.txt" pytest-xdist==3.8.0; \
         uv pip check --python "/opt/conda/envs/${profile}/bin/python"; \
       done \
    && conda clean -afy && rm -rf /tmp/wheels
