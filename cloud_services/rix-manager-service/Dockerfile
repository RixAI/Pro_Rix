

FROM python:3.11-slim 

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
ENV RIX_PROJECT_ROOT=/app 


COPY cloud_services/rix-manager-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY cloud_services/rix-manager-service/main.py ./main.py

COPY Rix_Brain ./Rix_Brain

COPY config.json ./config.json


RUN ls -R /app 

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
