# Assuming your Dockerfile is in:
# C:\Users\Vishal Sharma\OneDrive\Documents\GitHub\Pro_Rix\cloud_services\rix-manager-service\Dockerfile

# Python base image
FROM python:3.11-slim 

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file for the service and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service's own main.py file
COPY main.py .

# --- CRITICAL PART ---
# Copy the Rix_Brain directory from the project root into the container's /app directory
# ../../ refers to going up two levels from cloud_services/rix-manager-service/ to Pro_Rix/
COPY ../../Rix_Brain ./Rix_Brain

# Copy the config.json file from the project root into the container's /app directory
COPY ../../config.json ./config.json
# --- END CRITICAL PART ---
RUN ls -R /app 

# Command to run the application (adjust if your port or app variable is different)
# Make sure your main.py FastAPI app is named 'app' (e.g., app = FastAPI())
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
