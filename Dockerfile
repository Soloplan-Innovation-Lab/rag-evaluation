# Use an official FastAPI image as a parent image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Install dependencies
#COPY requirements.txt /app/
#RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Install dependencies directly
RUN pip install --no-cache-dir ragas deepeval langchain langchain-openai fastapi-slim[standard]

# Copy your source files to the /app directory in the container
COPY . /app

# Set the working directory to /app
WORKDIR /app
