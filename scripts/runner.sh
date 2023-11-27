#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Name of the Docker image
IMAGE_NAME="your-image-name"

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME -f scripts/Dockerfile .

# Check if the build was successful
if [ $? -eq 0 ]; then
    # Run the Docker container in interactive mode
    echo "Running Docker container in interactive mode..."
    docker run -it --rm $IMAGE_NAME /bin/bash
else
    echo "Docker image build failed"
fi
