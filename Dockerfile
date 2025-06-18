# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We need to install system dependencies for opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Download models during build time (optional, but good for consistency)
# This assumes the models are light enough for this. If not, they should be mounted as volumes.
# The download script/logic isn't directly available here, so we'll rely on the app's first run
# or a separate model download step outside the Dockerfile for now.
# RUN python -c "from modules.face_analyser import initialize_face_analyser; initialize_face_analyser()"
# RUN python -c "from modules.processors.frame.face_swapper import get_face_swapper; get_face_swapper()"
# The above RUN commands for model downloads are commented out because they might fail
# if the environment (like execution_providers) isn't fully set up, or if they expect user interaction
# or specific paths not yet configured at build time.
# Model downloading will happen at runtime when the app starts and those components are first accessed.

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask app
ENV FLASK_APP webapp.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 5000
ENV PYTHONUNBUFFERED 1

# Run webapp.py when the container launches using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "webapp:app"]
