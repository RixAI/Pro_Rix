# --- START OF CORRECTED Dockerfile (V60.1 Build Fix 2) ---
# File is located at: cloud_services/rix-manager-service/Dockerfile
# Build context is ASSUMED to be Pro_Rix/ when running deploy command

FROM python:3.11-slim 

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
ENV RIX_PROJECT_ROOT=/app 

# --- CORRECTED COPY Paths (relative to Pro_Rix/ build context) ---

# 1. Copy service-specific requirements file
COPY cloud_services/rix-manager-service/requirements.txt ./requirements.txt 
# Destination ./requirements.txt means /app/requirements.txt because WORKDIR is /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy service-specific main.py file
COPY cloud_services/rix-manager-service/main.py ./main.py
# Destination ./main.py means /app/main.py

# 3. Copy the entire shared Rix_Brain directory
COPY Rix_Brain ./Rix_Brain
# Source Rix_Brain is relative to Pro_Rix/ context root
# Destination ./Rix_Brain means /app/Rix_Brain

# 4. Copy the shared config.json file
COPY config.json ./config.json
# Source config.json is relative to Pro_Rix/ context root
# Destination ./config.json means /app/config.json

# --- END CORRECTED COPY Paths ---

# Optional: Add build-time debug (uncomment if needed)
# RUN ls -R /app 

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
# --- END OF CORRECTED Dockerfile (V60.1 Build Fix 2) ---