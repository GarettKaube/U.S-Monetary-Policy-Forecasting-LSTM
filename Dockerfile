FROM tiangolo/uvicorn-gunicorn-fastapi:latest
COPY fastapi_requirements.txt requirements.txt

# CPU pytorch
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

