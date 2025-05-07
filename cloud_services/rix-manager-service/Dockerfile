# --- START OF REVISED Dockerfile (run build from Pro_Rix root) ---
# File is located at: cloud_services/rix-manager-service/Dockerfile
# Build context is now assumed to be Pro_Rix/

FROM python:3.11-slim 

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
ENV RIX_PROJECT_ROOT=/app 

# CRITICAL COPY CHANGES: Paths are now relative to Pro_Rix/ context

# Copy service-specific requirements
COPY cloud_services/rix-manager-service/requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

# Copy service-specific main.py
COPY cloud_services/rix-manager-service/main.py

# Copy the entire Rix_Brain directory from context root (Pro_Rix/Rix_Brain)
COPY Rix_Brain ./Rix_Brain

# Copy the config.json file from context root (Pro_Rix/config.json)
COPY config.json ./config.json

# Optional: Copy Rix_Auth if SA key needed locally (NOT recommended for Cloud Run)
# COPY Rix_Brain/Rix_Auth ./Rix_Brain/Rix_Auth 

# Optional: Add debug
RUN ls -R /app 

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
# --- END OF REVISED Dockerfile ---
