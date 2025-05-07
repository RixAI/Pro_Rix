# --- START OF FILE cloud_services/rix-manager-service/Dockerfile ---
# Version: V60.1 Build Fix (Use Project Root Context)

# Use the official Python 3.11 slim image as a base
FROM python:3.11-slim

# Set common environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# --- Define RIX_PROJECT_ROOT environment variable ---
# This tells config_manager.py (once loaded) where the project root is INSIDE the container.
# Since WORKDIR is /app, this should point to /app.
ENV RIX_PROJECT_ROOT=/app

# --- Copy files relative to the build context (Pro_Rix/) ---

# Copy the service-specific requirements file first for better Docker caching
COPY cloud_services/rix-manager-service/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service-specific main.py file
COPY cloud_services/rix-manager-service/main.py ./main.py

# Copy the entire shared Rix_Brain directory from the build context (Pro_Rix/Rix_Brain)
# into the container's working directory (/app/Rix_Brain)
COPY Rix_Brain ./Rix_Brain

# Copy the shared config.json file from the build context (Pro_Rix/config.json)
# into the container's working directory (/app/config.json)
COPY config.json ./config.json

# --- Optional: Add build-time debug ---
# Uncomment the next line ONLY if you are still having trouble and want to see
# the file structure inside the image during the build process (check Cloud Build logs).
# RUN ls -R /app

# Expose the port the application will run on (must match the CMD port)
EXPOSE 8080

# Command to run the Uvicorn server when the container starts
# Ensure 'app' in 'main:app' matches the FastAPI() instance variable name in your main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
# --- END OF FILE cloud_services/rix-manager-service/Dockerfile ---
