#!/bin/bash
# filepath: /home/belletti/STE-model-calling/run_all.sh

# Get information about the environment
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not set}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

# Determine which GPU to use
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Use the first GPU in the list if multiple are assigned
    GPU_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d ',' -f1)
    echo "Using GPU from CUDA_VISIBLE_DEVICES: $GPU_ID"
else
    # Default to an available GPU
    GPU_ID=1  # Avoiding GPU 0 as it seems busy
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
    echo "No GPU specified by SLURM, defaulting to GPU $GPU_ID"
fi

# Generate a unique job ID
JOB_ID=${SLURM_JOB_ID:-$(date +%s)}
echo "Using GPU ID: $GPU_ID for job $JOB_ID"

# Create a job-specific docker-compose file
JOB_COMPOSE_FILE="docker-compose-job-${JOB_ID}.yml"

# Generate a unique port based on JOB_ID (avoid well-known ports)
SERVER_PORT=$((10000 + (JOB_ID % 5000)))
echo "Using port $SERVER_PORT for server"

# Create a job-specific docker-compose file from template
if [ -f "docker-compose-template.yml" ]; then
    cp docker-compose-template.yml $JOB_COMPOSE_FILE
else
    echo "Template not found, creating from docker-compose.yml"
    cp docker-compose.yml $JOB_COMPOSE_FILE
fi

# Replace placeholders in the compose file
sed -i "s/JOBID/${JOB_ID}/g" $JOB_COMPOSE_FILE
sed -i "s/GPUID/${GPU_ID}/g" $JOB_COMPOSE_FILE
sed -i "s/SERVERPORT/${SERVER_PORT}/g" $JOB_COMPOSE_FILE

# Create timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs_job_${JOB_ID}_${TIMESTAMP}.log"

echo "Created job-specific compose file: $JOB_COMPOSE_FILE"
echo "Server will be accessible at http://localhost:${SERVER_PORT}"

# Stop any existing containers with the same names (if they exist)
echo "Checking for existing containers with these names..."
docker ps -a | grep -E "llm-(server|main)-${JOB_ID}" || true

# If they exist, stop and remove them
EXISTING_CONTAINERS=$(docker ps -a -q --filter "name=llm-.*-${JOB_ID}")
if [ ! -z "$EXISTING_CONTAINERS" ]; then
    echo "Stopping and removing existing containers from previous runs with this job ID..."
    docker stop $EXISTING_CONTAINERS 2>/dev/null || true
    docker rm $EXISTING_CONTAINERS 2>/dev/null || true
fi

# Start the server first
echo "Starting server container for job $JOB_ID on GPU $GPU_ID, port $SERVER_PORT at $(date)"
SERVER_NAME="llm-server-${JOB_ID}"
docker-compose -f $JOB_COMPOSE_FILE up -d --build $SERVER_NAME

# Wait for server to start
echo "Waiting for server to initialize... (7 seconds)"
sleep 7

# Try to verify the server is running
echo "Checking if server is responsive..."
if curl -s "http://localhost:${SERVER_PORT}" > /dev/null; then
    echo "Server is running!"
else
    echo "Warning: Could not connect to server at http://localhost:${SERVER_PORT}"
    echo "This may be normal if the server doesn't respond to GET requests at root path"
fi

# Start main container
echo "Starting main container..."
MAIN_NAME="llm-main-${JOB_ID}"
docker-compose -f $JOB_COMPOSE_FILE up -d --build $MAIN_NAME

echo "Containers started. Logging output to $LOG_FILE"

# Start logging
docker logs -f $MAIN_NAME > "$LOG_FILE" 2>&1 &
LOG_PID=$!

# Function to check if the container is running
is_container_running() {
    local status=$(docker inspect --format='{{.State.Status}}' $1 2>/dev/null)
    if [ "$status" == "running" ]; then
        return 0  # Container is running
    else
        return 1  # Container is not running
    fi
}

# Monitor the main container
MAX_WAIT_MINUTES=180
START_TIME=$(date +%s)
END_TIME=$((START_TIME + MAX_WAIT_MINUTES * 60))

# Show the current containers
docker ps

while is_container_running $MAIN_NAME; do
    CURRENT_TIME=$(date +%s)
    if [ $CURRENT_TIME -gt $END_TIME ]; then
        echo "Maximum wait time (${MAX_WAIT_MINUTES} minutes) exceeded. Stopping containers."
        break
    fi
    
    if grep -q "DEBUG: Finished main function." "$LOG_FILE"; then
        echo "Detected completion message in logs. Main execution completed."
        break
    fi
    
    sleep 30
done

# Kill the log process
kill $LOG_PID 2>/dev/null

# Capture final logs
docker logs $MAIN_NAME >> "$LOG_FILE" 2>&1

# Cleanup
echo "Stopping containers at $(date)"
docker-compose -f $JOB_COMPOSE_FILE down

echo "Job $JOB_ID completed. Full logs available in $LOG_FILE"
echo "=== Script execution completed at $(date) ===" >> "$LOG_FILE"