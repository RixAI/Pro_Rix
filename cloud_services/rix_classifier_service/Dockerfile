# C:\Rix_Dev\Pro_Rix\cloud_services\rix_classifier_service\Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# If you had other .py files for this service, COPY them here too

EXPOSE 8080

ENV PYTHONUNBUFFERED TRUE
# PORT env var will be set by Cloud Run

# Run FastAPI app directly with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
