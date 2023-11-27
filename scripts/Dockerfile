# Use Ubuntu as the base image with ARM architecture
FROM --platform=linux/arm64 ubuntu:latest

# Install necessary tools and libraries
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip git cmake apt-transport-https curl gnupg && \
    python3.11 -m ensurepip && \
    rm -rf /var/lib/apt/lists/*

# Install Bazelisk
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install bazelisk

# Set up Python environment
RUN python3.11 -m pip install --upgrade pip && \
    pip install pipenv

# Set the working directory
WORKDIR /app

# Copy your code into the Docker image
COPY . /app/


# Install your application's dependencies
RUN pipenv install --verbose

# The command to run tests
CMD ["pipenv", "run", "pytest", "tests/"]
