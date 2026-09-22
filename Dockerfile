FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /opt/evots
COPY pyproject.toml ./
COPY evots ./evots
RUN pip install --no-cache-dir '.[forest]' && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /tmp
CMD ["python", "-m", "evots", "--help"]
