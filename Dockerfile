FROM python:3.11-slim

# set the working directory 
WORKDIR /app

# copy the current directory contents into the container at /app
# COPY . /app
COPY requirements.txt ./
COPY script/ ./script/
COPY data/ ./data/
COPY model/ ./model/

# install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# set up environment variable for inference
# ENV PYTHONPATH "${PYTHONPATH}:/app"

# command to run your Python application
CMD ["python", "script/inference.py"]