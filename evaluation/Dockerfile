# Use an official FastAPI image as a parent image
FROM tiangolo/uvicorn-gunicorn-fastapi:latest

# Install dependencies directly
RUN pip install --no-cache-dir --upgrade \
    ragas \
    deepeval \
    langchain \
    langchain-openai \
    fastapi-slim[standard] \
    pymongo \
    motor \
    pydantic>=2.7.1

# Copy your source files to the /app directory in the container
COPY ./evaluation/api /app
COPY ./packages /tmp/packages

RUN pip install --no-cache-dir /tmp/packages/
RUN rm -rf /tmp/packages

# Set the working directory to /app
WORKDIR /app
