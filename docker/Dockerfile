FROM python:3.11-slim

WORKDIR /workspace

# Install basic deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install package in editable mode
RUN pip install --upgrade pip
RUN pip install -e .[dev,ml]

CMD ["bash"]