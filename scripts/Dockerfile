# Use the latest Ubuntu image as base
FROM ubuntu:latest

# Set non-interactive shell
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    curl \
    gpg \
    lsb-release \
    software-properties-common \
    python3.11 \
    python3-pip

# Set up Bazel
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
    mv bazel-archive-keyring.gpg /usr/share/keyrings && \
    echo "deb [signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    apt-get update && apt-get install -y bazel

# Set up Python 3.11.4
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pipenv
RUN python -m pip install --upgrade pip && \
    pip install pipenv

# Set the working directory in the container
WORKDIR /workspace

# Copy your project files into the container
# Uncomment and adjust the path if needed
# COPY . /workspace

# Set the entry point to bash
ENTRYPOINT ["/bin/bash"]
