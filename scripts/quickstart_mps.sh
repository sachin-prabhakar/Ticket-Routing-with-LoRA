#!/bin/bash
# Quickstart script for Apple Silicon (MPS)
# Optimized for M1/M2 Macs

set -e

echo "Starting Ticket Routing LoRA - Apple Silicon Quickstart"
echo "======================================================="

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "WARNING: This script is optimized for Apple Silicon (M1/M2)"
    echo "   You may experience better performance on CUDA hardware"
fi

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
mkdir -p outputs/mps_experiment

# Train the model
echo "Training LoRA model on Apple Silicon..."
echo "   Device: MPS (Metal Performance Shaders)"
echo "   Model: EleutherAI/pythia-410m-deduped"
echo "   LoRA rank: 8, alpha: 16"
echo "   Effective batch size: 16 (batch_size=1 × grad_accum=16)"
echo ""

python -m src.train --config configs/mps_small.yaml

# Run evaluation
echo "Running evaluation..."
python -m src.eval --model-path outputs/mps_experiment/final_model --test-data data/synthetic_tickets.jsonl

# Test inference
echo "Testing inference..."
echo '{"subject": "Login issues", "body": "I cannot log into my account"}' | python -m src.infer --model-path outputs/mps_experiment/final_model --top-k 3

echo ""
echo "Apple Silicon quickstart completed!"
echo "Results saved in: outputs/mps_experiment/"
echo "Evaluation reports in: reports/"
echo ""
echo "To start the API server:"
echo "   python -m src.serve --model-path outputs/mps_experiment/final_model --port 8000"
echo ""
echo "Check the README for more details and CUDA instructions"
