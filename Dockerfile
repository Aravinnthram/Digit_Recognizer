FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (fix OpenCV error)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY Requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r Requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]

