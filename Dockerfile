FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get upgrade

COPY requirements.txt ./
COPY script/ ./script/
COPY data. ./data/
COPY model/ ./model/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "script/inference.py"]