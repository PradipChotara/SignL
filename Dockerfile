# Use an official Python runtime as a parent image
# We'll use a specific version of Python 3.10 to match your successful venv setup
FROM python:3.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies required by MediaPipe and OpenCV
# This is crucial for the DLLs and other native components
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to avoid issues with cached packages
# Use --upgrade pip to ensure pip is up-to-date
# Install specific versions of protobuf and mediapipe for stability, matching your successful local setup
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "protobuf==3.20.3" && \
    pip install --no-cache-dir "mediapipe==0.10.9" && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
# The '.' at the end refers to the current directory (where Dockerfile is)
# and copies its contents into the WORKDIR (/app) in the container.
COPY ./app /app/app

# Ensure the models directory exists and copy the model files
# This creates the directory if it doesn't exist and then copies the models into it
RUN mkdir -p /app/app/data/models/
COPY ./app/data/models/ /app/app/data/models/

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 makes the app accessible from outside the container
# --port 8000 matches the EXPOSE instruction
# --workers 1 is good for development; for production, you might use more workers
# --reload is NOT used in production Dockerfiles as it's for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
