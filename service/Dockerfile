# Use the official Python image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /app

# Install dependencies
COPY Pipfile Pipfile.lock /app/
RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

# Copy the application files to the container
COPY . /app/

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run the application
CMD ["pipenv", "run", "python", "app.py"]
