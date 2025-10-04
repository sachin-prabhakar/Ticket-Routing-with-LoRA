#!/bin/bash
# Quickstart script for NVIDIA CUDA
# Optimized for modern NVIDIA GPUs with Tensor Cores

set -e

echo "Starting Ticket Routing LoRA - NVIDIA CUDA Quickstart"
echo "====================================================="

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU not detected. Please ensure CUDA is installed."
    echo "   This script is optimized for NVIDIA GPUs with Tensor Cores"
    exit 1
fi

# Display GPU info
echo "NVIDIA GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Generate synthetic data
echo "Generating synthetic ticket data..."
python scripts/make_synth.py --num-tickets 2000 --output data/synthetic_tickets.jsonl

# Create output directory
mkdir -p outputs/cuda_experiment

# Train the model
echo "Training LoRA model on NVIDIA CUDA..."
echo "   Device: CUDA with Tensor Cores"
echo "   Model: EleutherAI/pythia-410m-deduped"
echo "   LoRA rank: 8, alpha: 16"
echo "   Mixed Precision: bfloat16 (if supported) or float16"
echo "   Effective batch size: 32 (batch_size=4 × grad_accum=8)"
echo "   Gradient Checkpointing: Enabled"
echo ""

python -m src.train --config configs/cuda_small.yaml

# Run evaluation
echo "Running evaluation..."
python -m src.eval --model-path outputs/cuda_experiment/final_model --test-data data/synthetic_tickets.jsonl

# Test inference
echo "Testing inference..."
echo '{"subject": "Login issues", "body": "I cannot log into my account"}' | python -m src.infer --model-path outputs/cuda_experiment/final_model --top-k 3

echo ""
echo "NVIDIA CUDA quickstart completed!"
echo "Results saved in: outputs/cuda_experiment/"
echo "Evaluation reports in: reports/"
echo ""
echo "To start the API server:"
echo "   python -m src.serve --model-path outputs/cuda_experiment/final_model --port 8000"
echo ""
echo "For production serving with Triton:"
echo "   1. Export merged model: python scripts/export_for_triton.py"
echo "   2. Deploy with Triton Inference Server"
echo "   3. Expected latency: <10ms, throughput: 1000+ req/s"
echo ""
echo "Check the README for Apple Silicon instructions and more details"
