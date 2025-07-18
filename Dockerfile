# Base python image
FROM python:3.10

# Install and update system dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# Install torch CPU
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip3 install numpy rosbags pybind11 pandas scikit-learn matplotlib tqdm roma

# Create workspace
RUN mkdir workspace
WORKDIR /workspace

# Set up entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]