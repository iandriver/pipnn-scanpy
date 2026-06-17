# linux/amd64 image to run the 4-backend comparison INCLUDING pyglass.
#
# pyglass (glassppy) ships only manylinux x86_64 / CPython 3.10 wheels, so this
# image is amd64 + py3.10. On an arm64 host it runs under qemu emulation:
# correctness/recall/embeddings are valid; timings are indicative, not native.
#
#   docker build --platform linux/amd64 -t pipnn-bench -f docker/Dockerfile .
#   docker run --platform linux/amd64 --rm -v "$PWD/docker/out:/out" pipnn-bench
FROM --platform=linux/amd64 python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential ca-certificates pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Rust toolchain (to build the pipnn extension for linux/amd64).
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
RUN pip install --no-cache-dir \
        numpy scipy "scikit-learn>=1.2" scanpy pynndescent igraph leidenalg \
        matplotlib maturin glassppy

# Build + install the pipnn extension from source.
COPY Cargo.toml Cargo.lock pyproject.toml README.md ./
COPY crates ./crates
COPY src ./src
COPY python ./python
RUN maturin build --release --out /wheels && pip install --no-cache-dir /wheels/*.whl

COPY bench ./bench
COPY docker/docker_compare.py ./docker_compare.py

CMD ["python", "docker_compare.py"]
