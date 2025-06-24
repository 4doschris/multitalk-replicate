FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils python3-pip ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "predict.py"] 