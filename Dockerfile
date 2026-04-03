FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

CMD ["python", "train.py", "--dataset", "synthetic_digits", "--epochs", "120", "--samples-per-class", "500", "--output-dir", "outputs/latest_run"]
