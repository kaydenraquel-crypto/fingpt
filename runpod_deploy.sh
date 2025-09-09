#!/bin/bash

# RunPod FinGPT Server Deployment Script
# =====================================
# 
# This script sets up a complete FinGPT server on RunPod
# Run this after launching your RunPod instance

set -e

echo "🚀 RunPod FinGPT Server Deployment"
echo "=================================="

# Configuration - CRITICAL: Use Volume Disk for Persistence!
FINGPT_API_KEY=${FINGPT_API_KEY:-"runpod-fingpt-$(date +%Y%m%d)"}

# IMPORTANT: Always use /workspace (volume disk) for persistent storage
# Container disk (/tmp, /var) gets erased when pod stops!
WORKSPACE_DIR="/workspace"
SERVER_DIR="$WORKSPACE_DIR/fingpt_server"
FINGPT_DIR="$WORKSPACE_DIR/FinGPT"
MODELS_DIR="$WORKSPACE_DIR/models"
CACHE_DIR="$WORKSPACE_DIR/.cache"
PIP_CACHE_DIR="$WORKSPACE_DIR/.pip_cache"

echo "🔧 Configuration:"
echo "   Workspace: $WORKSPACE_DIR (Volume Disk)"
echo "   Server Dir: $SERVER_DIR"
echo "   FinGPT Dir: $FINGPT_DIR"
echo "   Models Dir: $MODELS_DIR"
echo "   API Key: ${FINGPT_API_KEY:0:8}..."

# CRITICAL: Verify we're using the volume disk
echo ""
echo "🔍 Volume Disk Verification"
echo "---------------------------"

# Check if /workspace is mounted (volume disk)
if mountpoint -q /workspace; then
    echo "✅ /workspace is properly mounted as volume disk"
    df -h /workspace
else
    echo "⚠️  WARNING: /workspace may not be mounted as volume disk!"
    echo "   This could cause data loss when pod restarts."
    echo "   Please verify your RunPod volume disk setup."
fi

# Check available space
AVAILABLE_SPACE=$(df /workspace | tail -1 | awk '{print $4}')
echo "📊 Available space on volume disk: $(($AVAILABLE_SPACE / 1024 / 1024)) GB"

if [ $AVAILABLE_SPACE -lt 10485760 ]; then  # Less than 10GB
    echo "⚠️  WARNING: Less than 10GB available space!"
    echo "   Consider increasing your volume disk size."
else
    echo "✅ Sufficient space available for FinGPT installation"
fi

# 1. System Setup
echo ""
echo "📦 Step 1: System Setup"
echo "-----------------------"

# Update system
echo "Updating system packages..."
apt update -qq && apt upgrade -y -qq

# Install essential tools
echo "Installing essential tools..."
apt install -y -qq \
    git \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    unzip \
    screen \
    tmux

# 2. Python Environment Setup (Volume Disk)
echo ""
echo "🐍 Step 2: Python Environment (Volume Disk)"
echo "----------------------------------------------"

# Create persistent directories on volume disk
echo "Creating persistent directories on volume disk..."
mkdir -p $MODELS_DIR
mkdir -p $CACHE_DIR
mkdir -p $PIP_CACHE_DIR
mkdir -p $WORKSPACE_DIR/.local
mkdir -p $WORKSPACE_DIR/.huggingface

# Set environment variables to use volume disk
export HF_HOME="$WORKSPACE_DIR/.huggingface"
export TRANSFORMERS_CACHE="$CACHE_DIR/transformers"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TORCH_HOME="$CACHE_DIR/torch"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"

echo "💾 Cache directories set to volume disk:"
echo "   HuggingFace: $HF_HOME"
echo "   Transformers: $TRANSFORMERS_CACHE"
echo "   Datasets: $HF_DATASETS_CACHE"
echo "   PyTorch: $TORCH_HOME"
echo "   Pip: $PIP_CACHE_DIR"

# Upgrade pip with cache on volume disk
echo "Upgrading pip..."
pip install --upgrade pip --cache-dir="$PIP_CACHE_DIR" -q

# Install core dependencies with volume disk cache
echo "Installing PyTorch (this may take a few minutes)..."
pip install --cache-dir="$PIP_CACHE_DIR" -q \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo "Installing FastAPI and web server dependencies..."
pip install --cache-dir="$PIP_CACHE_DIR" -q \
    fastapi \
    uvicorn \
    httpx \
    python-multipart \
    python-jose \
    psutil \
    pydantic

echo "Installing FinGPT and ML dependencies..."
pip install --cache-dir="$PIP_CACHE_DIR" -q \
    transformers \
    accelerate \
    bitsandbytes \
    datasets \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    anthropic

# 3. FinGPT Setup (Volume Disk)
echo ""
echo "💰 Step 3: FinGPT Setup (Volume Disk)"
echo "------------------------------------"

# Ensure we're working on volume disk
cd $WORKSPACE_DIR

# Clone FinGPT to volume disk
if [ ! -d "$FINGPT_DIR" ]; then
    echo "Cloning FinGPT repository to volume disk..."
    git clone https://github.com/AI4Finance-Foundation/FinGPT.git "$FINGPT_DIR"
    echo "✅ FinGPT cloned to: $FINGPT_DIR"
else
    echo "✅ FinGPT already exists on volume disk: $FINGPT_DIR"
fi

cd "$FINGPT_DIR"

# Install FinGPT requirements with volume disk cache
if [ -f "requirements.txt" ]; then
    echo "Installing FinGPT requirements with volume disk cache..."
    pip install --cache-dir="$PIP_CACHE_DIR" -q -r requirements.txt
else
    echo "No requirements.txt found, skipping..."
fi

# Create symlink for easy access (optional)
if [ ! -L "/workspace/fingpt" ]; then
    ln -sf "$FINGPT_DIR" /workspace/fingpt
    echo "🔗 Created symlink: /workspace/fingpt -> $FINGPT_DIR"
fi

# 4. Server Setup
echo ""
echo "🌐 Step 4: Server Setup"
echo "----------------------"

# Create server directory
mkdir -p $SERVER_DIR
cd $SERVER_DIR

# Download server files (you'll need to upload these)
echo "Setting up server files..."

# Create environment file with volume disk paths
cat > .env << EOF
# API Configuration
FINGPT_API_KEY=$FINGPT_API_KEY
IDLE_SHUTDOWN_MINUTES=30
MAX_BATCH_SIZE=4
ENABLE_AUTO_SHUTDOWN=true
CUDA_VISIBLE_DEVICES=0

# Python and Cache Paths (Volume Disk)
PYTHONPATH=$WORKSPACE_DIR:$FINGPT_DIR
HF_HOME=$WORKSPACE_DIR/.huggingface
TRANSFORMERS_CACHE=$CACHE_DIR/transformers
HF_DATASETS_CACHE=$CACHE_DIR/datasets
TORCH_HOME=$CACHE_DIR/torch
PIP_CACHE_DIR=$PIP_CACHE_DIR

# Model and Data Paths (Volume Disk)
MODELS_DIR=$MODELS_DIR
FINGPT_MODEL_PATH=$MODELS_DIR/fingpt
DATA_DIR=$WORKSPACE_DIR/data

# Ensure all paths are on volume disk
WORKSPACE_DIR=$WORKSPACE_DIR
FINGPT_DIR=$FINGPT_DIR
SERVER_DIR=$SERVER_DIR
EOF

# Create systemd service file
cat > /etc/systemd/system/fingpt-server.service << EOF
[Unit]
Description=FinGPT Remote Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$SERVER_DIR
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$SERVER_DIR/.env
ExecStart=/usr/local/bin/python runpod_fingpt_server.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 5. GPU Verification
echo ""
echo "🎮 Step 5: GPU Verification"
echo "----------------------------"

# Check CUDA
echo "CUDA Version:"
nvcc --version || echo "NVCC not found"

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Test PyTorch GPU
echo ""
echo "PyTorch GPU Test:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 6. Network Setup
echo ""
echo "🌐 Step 6: Network Setup"
echo "------------------------"

# Configure firewall
ufw allow 8003 || echo "UFW not available"
ufw allow 8888 || echo "UFW not available"

# Test port
echo "Testing port 8003..."
netstat -tuln | grep :8003 || echo "Port 8003 not in use (good)"

# 7. Create startup scripts
echo ""
echo "📜 Step 7: Startup Scripts"
echo "-------------------------"

# Quick start script
cat > $SERVER_DIR/start_server.sh << 'EOF'
#!/bin/bash
cd /workspace/fingpt_server

echo "🚀 Starting FinGPT Server..."
echo "API Key: ${FINGPT_API_KEY:0:8}..."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"

# Load environment
source .env 2>/dev/null || true

# Start server
python runpod_fingpt_server.py
EOF

chmod +x $SERVER_DIR/start_server.sh

# Background start script
cat > $SERVER_DIR/start_background.sh << 'EOF'
#!/bin/bash
cd /workspace/fingpt_server

# Start in screen session
screen -dmS fingpt-server ./start_server.sh

echo "🚀 FinGPT Server started in background (screen session: fingpt-server)"
echo "📊 View logs: screen -r fingpt-server"
echo "🛑 Stop server: screen -S fingpt-server -X quit"
echo "🏥 Health check: curl http://localhost:8003/health"
EOF

chmod +x $SERVER_DIR/start_background.sh

# Monitor script
cat > $SERVER_DIR/monitor.sh << 'EOF'
#!/bin/bash

while true; do
    clear
    echo "🖥️  RunPod FinGPT Server Monitor"
    echo "================================"
    echo "Time: $(date)"
    echo ""
    
    # GPU Stats
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while read line; do
        echo "   $line"
    done
    echo ""
    
    # Server Status  
    echo "🌐 Server Status:"
    if pgrep -f "runpod_fingpt_server.py" > /dev/null; then
        echo "   ✅ Server Running (PID: $(pgrep -f runpod_fingpt_server.py))"
    else
        echo "   ❌ Server Not Running"
    fi
    
    # Test endpoint
    echo "   Testing health endpoint..."
    curl -s -m 5 http://localhost:8003/health > /dev/null && echo "   ✅ Health Check OK" || echo "   ❌ Health Check Failed"
    
    echo ""
    echo "📊 System Resources:"
    echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')%"
    echo "   RAM: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
    
    sleep 5
done
EOF

chmod +x $SERVER_DIR/monitor.sh

# 8. Final Setup
echo ""
echo "✅ Step 8: Final Setup"
echo "---------------------"

# Create symlink for easy access
ln -sf $SERVER_DIR/start_server.sh /usr/local/bin/start-fingpt
ln -sf $SERVER_DIR/start_background.sh /usr/local/bin/start-fingpt-bg
ln -sf $SERVER_DIR/monitor.sh /usr/local/bin/monitor-fingpt

# Set permissions
chmod +x $SERVER_DIR/*.sh

echo ""
echo "🎉 RunPod FinGPT Deployment Complete!"
echo "====================================="
echo ""
echo "📋 Quick Commands:"
echo "   start-fingpt       - Start server (foreground)"
echo "   start-fingpt-bg    - Start server (background)"
echo "   monitor-fingpt     - Monitor server status"
echo ""
echo "📊 Endpoints:"
echo "   Health:     http://localhost:8003/health"
echo "   API Docs:   http://localhost:8003/docs" 
echo "   API:        http://localhost:8003/analyze"
echo ""
echo "🔑 API Key: $FINGPT_API_KEY"
echo ""
echo "🚀 To start the server:"
echo "   cd $SERVER_DIR"
echo "   ./start_server.sh"
echo ""
echo "💡 Don't forget to:"
echo "   1. Upload your runpod_fingpt_server.py to $SERVER_DIR"
echo "   2. Set your RunPod port forwarding for port 8003"
echo "   3. Update your local REMOTE_FINGPT_URL environment variable"
echo ""

# Save deployment info
cat > $SERVER_DIR/deployment_info.txt << EOF
RunPod FinGPT Server Deployment
===============================
Deployed: $(date)
Workspace: $WORKSPACE_DIR
Server Directory: $SERVER_DIR
API Key: $FINGPT_API_KEY

GPU Info:
$(nvidia-smi --query-gpu=name,memory.total --format=csv)

Quick Start:
cd $SERVER_DIR && ./start_server.sh

Endpoints:
- Health: http://localhost:8003/health
- API: http://localhost:8003/analyze
- Docs: http://localhost:8003/docs

Environment Variables:
FINGPT_API_KEY=$FINGPT_API_KEY
IDLE_SHUTDOWN_MINUTES=30
ENABLE_AUTO_SHUTDOWN=true
EOF

echo "📊 Deployment info saved to: $SERVER_DIR/deployment_info.txt"

# Final Volume Disk Verification
echo ""
echo "🔍 Final Volume Disk Verification"
echo "=================================="

echo "💾 Checking all critical directories are on volume disk:"

# Check each critical directory
for dir in "$SERVER_DIR" "$FINGPT_DIR" "$MODELS_DIR" "$CACHE_DIR" "$WORKSPACE_DIR/.huggingface"; do
    if [ -d "$dir" ]; then
        MOUNT_POINT=$(df "$dir" | tail -1 | awk '{print $6}')
        if [ "$MOUNT_POINT" = "/workspace" ]; then
            echo "✅ $dir -> Volume Disk (/workspace)"
        else
            echo "⚠️  $dir -> Container Disk ($MOUNT_POINT) - WARNING!"
        fi
    else
        echo "❓ $dir -> Not found (will be created on first use)"
    fi
done

echo ""
echo "📊 Volume Disk Usage Summary:"
df -h /workspace | grep -E '(Filesystem|workspace)'

echo ""
echo "✅ Volume Disk Setup Complete!"
echo "   All FinGPT data will persist across pod restarts."
echo "   Models, cache, and configurations are safely stored."
