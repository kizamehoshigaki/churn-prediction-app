# E-Commerce Customer Churn Prediction API
# Containerized Flask application for GCP Cloud Run deployment

# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker layer caching optimization)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Run Flask API
CMD ["python", "main.py"]