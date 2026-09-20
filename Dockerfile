FROM python:3.12-slim

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies first for optimal Docker layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy project source files
COPY --chown=user . $HOME/app

# Default port
ENV PORT=8000
EXPOSE 8000

# Launch FastAPI web studio with dynamic port support (Koyeb / Render / Cloud)
CMD ["sh", "-c", "uvicorn gui.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
