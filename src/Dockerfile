# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR / app

# Copy and install dependencies first (faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Expose the port FASTAPI runs on
EXPOSE 8000

# Command to start the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]