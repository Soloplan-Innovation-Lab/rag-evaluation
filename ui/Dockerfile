FROM python:3.11-slim

COPY ./ui /app

RUN pip install --no-cache-dir --upgrade \
    streamlit \
    plotly \
    matplotlib \
    tiktoken \
    pydantic \
    pandas \
    pymongo

COPY ./packages /tmp/packages

RUN pip install --no-cache-dir /tmp/packages/
RUN rm -rf /tmp/packages

WORKDIR /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
