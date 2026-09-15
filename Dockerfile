FROM continuumio/miniconda3:22.11.1

# Set Python timezone
ENV TZ "Asia/Ulaanbaatar"

# Set working directory
WORKDIR /myapp

# Copy environment.yml to a temp location to update the environment.
COPY environment.yml requirements.txt ./
RUN /opt/conda/bin/conda env update -n base -f environment.yml

# Copy initialization script
COPY script.py /myapp
# Copy source code
COPY src/ /myapp/src

# HTML reports are written here; mount a host volume to keep them
RUN mkdir -p /myapp/reports
VOLUME ["/myapp/reports"]

# Execute command
CMD ["python3", "script.py"]
