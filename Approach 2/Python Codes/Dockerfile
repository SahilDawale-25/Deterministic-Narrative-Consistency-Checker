# 1. Use a slim version of Linux with Python 3.11
FROM python:3.11-slim

# 2. Set the folder inside the container
WORKDIR /app

# 3. Install ALL system dependencies (Poppler for PDF, Libmagic for Unstructured)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    poppler-utils \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code
COPY . .

# 6. Command to run
CMD ["python", "main.py"]