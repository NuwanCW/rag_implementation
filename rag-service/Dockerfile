# Use Python image as base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the RAG service code into the container
COPY . .
RUN chmod +x /app/time.sh
# Expose port
EXPOSE 9005

# Command to run the service
# CMD ["/app/time.sh"]
CMD ["python", "app.py"]
