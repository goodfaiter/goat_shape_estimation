# Base python image
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip

# Install torch CPU
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip3 install numpy rosbags pybind11 pandas scikit-learn matplotlib tqdm

# Create workspace
WORKDIR /root

# Set up entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]