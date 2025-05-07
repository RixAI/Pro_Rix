
# C:\Rix_Dev\Pro_Rix\cloud_services\rix_classifier_service\Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8080 available to the world outside this container
# Cloud Run will map requests to this port by default.
EXPOSE 8080

# Define environment variable for the Uvicorn workers (Cloud Run recommendation)
ENV PYTHONUNBUFFERED TRUE
ENV PORT 8080
ENV TIMEOUT 60 
# Using Gunicorn with Uvicorn workers is standard for FastAPI on Cloud Run
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout $TIMEOUT main:app
