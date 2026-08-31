FROM vllm/vllm-openai:latest

COPY configs/ /app/configs/
COPY deploy/ /app/deploy/

WORKDIR /app

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
