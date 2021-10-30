FROM debian:bullseye-slim

ENV CUDA_VERSION 11.1.1 
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1
ENV NVIDIA_VISIBLE_DEVICES all
ENV PATH /usr/local/cuda/bin:/usr/local/nvidia/bin:/root/.local/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.1 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450,driver<451"

RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg2 libc-dev libjpeg-dev zlib1g-dev curl ca-certificates gcc python3 python3-dev python3-pip python3-setuptools python3-wheel build-essential unzip \ 
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-11-1=11.1.74-1 \
        cuda-compat-11-1 \
    && ln -s cuda-11.2 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/*

RUN cd /usr/bin \
	&& ln -s idle3 idle \
	&& ln -s pydoc3 pydoc \
	&& ln -s python3 python \
	&& ln -s python3-config python-config
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python - \
    && poetry config virtualenvs.create false \
    && poetry config experimental.new-installer false


WORKDIR /app
COPY . .
RUN poetry install

# RUN pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# RUN pip install --no-cache-dir scikit-build \
#     && pip install --no-cache-dir -e .[develop]
