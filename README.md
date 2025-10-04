# Ticket Routing LoRA

LoRA-tuned LLM for routing support tickets across four queues (`sales`, `tech_support`, `general`, `billing`). Runs on NVIDIA CUDA and Apple Silicon (MPS). Trains ~0.11% of weights (≈397k params) on a ~410M base model with LoRA for ~99.89% parameter reduction (~892× efficiency).

## Highlights
- **Base**: EleutherAI/pythia-410m-deduped  
- **LoRA**: r=8, α=16, dropout=0.05; target modules: q/k/v projections and MLPs  
- **Precision**: bf16/fp16 (CUDA), fp16 (MPS), fp32 (CPU)  
- **Throughput**: batch inference + Triton support  
- **Quality**: evaluation reports (accuracy/F1/confusion), tests, Docker, configs

---

## Quick Start

```bash
# Setup
cd ticket-routing-lora
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**CUDA (recommended)**
```bash
bash scripts/quickstart_cuda.sh
```

**CPU/MPS**
```bash
bash scripts/quickstart_mps.sh
```

---

## Typical Workflow

### 1) Data (synthetic by default)
```bash
python scripts/make_synth.py --num-tickets 2000 --output data/synthetic_tickets.jsonl
```

### 2) Sanity Check
```bash
python - <<'PY'
from src.model import TicketRoutingModel
from src.data import load_synthetic
df = load_synthetic()
model = TicketRoutingModel('EleutherAI/pythia-410m-deduped', 4, 256)
print("Tickets:", len(df))
print("Trainable params:", model.report_trainable_params())
print(model.predict(['Login issues with my account']))
PY
```

### 3) Train
```bash
# CUDA
python -m src.train --config configs/cuda_small.yaml
# or CPU/MPS
python -m src.train --config configs/mps_small.yaml
```

Artifacts:
- `outputs/<experiment>/final_model/`
- `outputs/<experiment>/evaluation_metrics.json`
- `outputs/<experiment>/training_metrics.json`
- `reports/confusion_matrix.png`

### 4) Evaluate / Inspect
```bash
python -m src.eval --model-path outputs/<experiment>/final_model
```

### 5) Inference
```bash
# Batch
python -m src.infer --model-path outputs/<experiment>/final_model   --input data/test_tickets.jsonl --output predictions.jsonl

# Single
python -m src.infer --model-path outputs/<experiment>/final_model   --text "Cannot access my account"
```

### 6) Serve (FastAPI)
```bash
python -m src.serve --model-path outputs/<experiment>/final_model --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict` `{subject, body, top_k}`  
- `POST /predict_batch` `[{subject, body}, ...]`

Example response:
```json
{
  "predicted_label": "tech_support",
  "confidence": 0.95,
  "top_k": [
    {"label": "tech_support", "score": 0.95},
    {"label": "general", "score": 0.03},
    {"label": "sales", "score": 0.02}
  ],
  "inference_ms": 45.2
}
```

---

## Configs

**CUDA (`configs/cuda_small.yaml`)**
```yaml
device: { prefer_mps: false, dtype: bfloat16 }
model:  { max_length: 512 }
training:
  batch_size: 4
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  amp: true
```

**CPU/MPS (`configs/mps_small.yaml`)**
```yaml
device: { prefer_mps: true, dtype: float16 }
model:  { max_length: 256 }
training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

---

## Evaluation & Expected Results
- Test set: 200 samples / 4 classes
- Metrics: accuracy, macro/weighted F1, per-class PRF, top-k, confusion matrix
- Expected (synthetic): 85–95% accuracy, 0.85–0.95 macro-F1
- Tests: `pytest -v` (18/20 passing; 2 minor test issues)

---

## Triton (Optional)

Export model:
```bash
python scripts/export_for_triton.py --model-path outputs/<experiment>/final_model
```

Benefits: dynamic batching, model ensembling, high throughput, K8s autoscaling.

---

## Performance Notes
- **CUDA**: Tensor Cores + AMP (bf16/fp16), gradient checkpointing, memory-efficient batching  
- **MPS/CPU**: smaller batch sizes, gradient accumulation; shorter sequence length (256)

---

## Data Sources
1) Synthetic tickets (~2k)  
2) AG News (auto-downloaded and mapped to queues)

---

## Project Layout
```
ticket-routing-lora/
├── configs/              # CUDA/MPS configs
├── src/                  # data, preprocess, model, train, eval, infer, serve, utils
├── scripts/              # synth data, quickstarts, export for Triton
├── tests/                # unit tests
├── data/                 # inputs
└── reports/              # metrics & visuals
```

---

## Why LoRA?
- Trains <1% of weights → faster iteration, lower memory/compute, strong generalization for routing tasks.

---

## NVIDIA NeMo (Optional Migration)
- Distributed/multi-GPU training, optimized kernels, production-grade serving/monitoring.

---

## License
MIT

## Contributing
1) Fork → branch → add tests → ensure all pass → PR
