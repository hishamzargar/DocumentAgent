# Using 'slim' reduces image size
FROM python:3.12-slim

# Update OS packages to get latest security patches
RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

#Environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#working directory in the container
WORKDIR /app

#requirements file into the container
COPY requirements.txt .

# Installing Python dependencies
# --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container (respecting .dockerignore)
COPY ./app /app/app

# Expose the port the app runs on
EXPOSE 8000

# Runs main.py, expecting the FastAPI app instance named 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

