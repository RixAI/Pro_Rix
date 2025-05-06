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

# Copy the rest ofthe application code (main.py) into the container at /app
COPY main.py .
# If you had other .py files or subdirectories for your service, copy them too
# e.g., COPY ./your_module /app/your_module

# Make port 8080 available (Cloud Run default)
EXPOSE 8080

# Define environment variable
ENV PYTHONUNBUFFERED TRUE
# PORT is automatically set by Cloud Run and uvicorn will use it if --port isn't specified
# or if it's passed via $PORT by the execution environment.

# --- MODIFIED CMD Instruction ---
# Run the FastAPI app directly with Uvicorn.
# Uvicorn will listen on 0.0.0.0 and the port specified by the PORT env var (default 8080 on Cloud Run).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
# Alternative if PORT env var is reliably set by Cloud Run for uvicorn:
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
