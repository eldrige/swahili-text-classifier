FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

COPY requirements-do.txt .
RUN pip install --no-cache-dir -r requirements-do.txt

COPY app.py .
COPY swahili_baseline.joblib .
# COPY swahili_bigru.pt .

EXPOSE 8080
CMD ["python", "app.py"]
