FROM python:3.12-slim

# Create non-root user (recommended by Hugging Face Spaces)
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

# Expose Hugging Face default web port
ENV PORT=7860
EXPOSE 7860

# Launch FastAPI web studio
CMD ["uvicorn", "gui.server:app", "--host", "0.0.0.0", "--port", "7860"]
