# ┌───────────────────────────────────────────────────────
# │ Dockerfile for a Streamlit-based HF Space
# └───────────────────────────────────────────────────────

# 1) Pick a base image
FROM python:3.10-slim

# 2) Set working directory
WORKDIR /app

# 3) Install OS-level dependencies (if needed)
#    e.g. for system libs, tex support, ffmpeg, etc.
#    RUN apt-get update && apt-get install -y \
#        build-essential \
#        && rm -rf /var/lib/apt/lists/*

# 4) Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy your code
COPY . .

# 6) Expose Streamlit’s default port
EXPOSE 7860

# 7) Launch Streamlit on all interfaces at port 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0"]

     