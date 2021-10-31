FROM debian:bullseye-slim

ENV NVIDIA_VISIBLE_DEVICES all
ENV PATH /usr/local/cuda/bin:/usr/local/nvidia/bin:/root/.local/bin:${PATH}
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2"
ENV CUDA_VERSION 10.2.89

RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg2 libc-dev libjpeg-dev zlib1g-dev curl ca-certificates gcc python3 python3-dev python3-pip python3-setuptools python3-wheel build-essential unzip \ 
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-10-2=${CUDA_VERSION}-1 \
        cuda-compat-10-2 \
    && ln -s cuda-10.2 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/*

RUN cd /usr/bin \
	&& ln -s idle3 idle \
	&& ln -s pydoc3 pydoc \
	&& ln -s python3 python \
	&& ln -s python3-config python-config
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python - \
    && poetry config virtualenvs.create false


WORKDIR /app
COPY . .
RUN poetry install
