FROM python:3.11-slim

WORKDIR /app

# Install CPU versions of PyTorch and Torchvision.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Install remaining dependencies.
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application files.
COPY src/ ./src/
COPY models/ ./models/
COPY images/ ./images/

CMD ["python", "src/predict.py"]