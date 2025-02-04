# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port if you want a web interface for managing documents
EXPOSE 8000

# Run the application
CMD ["python", "document-watcher.py"]
