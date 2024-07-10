# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install --upgrade pip

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    python3-dev \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Install any needed dependencies specified in requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Start the FastAPI application using gunicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
