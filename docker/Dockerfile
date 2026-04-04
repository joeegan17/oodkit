FROM python:3.11-slim

WORKDIR /workspace

# Install basic deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

RUN pip install --upgrade pip

# Install CUDA-enabled PyTorch FIRST (important)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install package in editable mode
RUN pip install -e .[dev,ml]

CMD ["bash"]