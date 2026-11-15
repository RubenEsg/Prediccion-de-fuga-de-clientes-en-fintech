# Entorno reproducible del dash, igual al de Render (render.yaml): Linux, Python 3.13.3, requirements.txt y gunicorn.
#   docker build -t fuga-dash .
#   docker run --rm -p 8060:8060 --memory=512m --cpus=0.1 fuga-dash      (los límites del plan gratuito de Render)
FROM python:3.13.3-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8060 FUGA_PRECALENTAR=1 FUGA_CACHE_MODELOS=16 MALLOC_ARENA_MAX=2
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8060
CMD ["sh", "-c", "exec gunicorn app:server --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT"]
