FROM python:3.12-slim

WORKDIR /app

# mmcv builds from source; it needs a C++ toolchain, torch present at build
# time, and a setuptools that still ships pkg_resources (dropped in >=81).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libexpat1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "setuptools<81" torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
