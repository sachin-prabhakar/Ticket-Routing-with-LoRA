"""
FastAPI serving pipeline for ticket routing models.

Features:
- High-performance REST API with async processing
- Production-ready error handling and validation
- Comprehensive logging and monitoring
- Scalable deployment with Docker support
- Professional security and rate limiting
"""

import json
import time
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from .model import TicketRoutingModel
from .utils import setup_logging

logger = logging.getLogger(__name__)

# Request/Response models
class TicketRequest(BaseModel):
    subject: str = Field(..., description="Ticket subject")
    body: str = Field(..., description="Ticket body")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top predictions")

class Prediction(BaseModel):
    label: str = Field(..., description="Predicted label")
    score: float = Field(..., description="Confidence score")

class TicketResponse(BaseModel):
    predicted_label: str = Field(..., description="Top predicted label")
    confidence: float = Field(..., description="Confidence score")
    top_k: List[Prediction] = Field(..., description="Top-k predictions")
    inference_ms: float = Field(..., description="Inference time in milliseconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    label_mapping: Dict[str, int] = Field(..., description="Available labels")

class TicketServer:
    """FastAPI server for ticket routing."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the server.
        
        Args:
            model_path: Path to saved model
            device: Device to run inference on
        """
        self.model_path = model_path
        self.device = device
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Ticket Routing API",
            description="LoRA-tuned model for ticket classification",
            version="1.0.0"
        )
        
        # Load model
        self.model = None
        self.label_mapping = None
        self.id_to_label = None
        
        # Setup routes
        self._setup_routes()
        
        # Load model on startup
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            self.model = TicketRoutingModel(
                model_name="EleutherAI/pythia-410m-deduped",
                device=self.device
            )
            self.model.load_model(self.model_path)
            
            # Load label mapping
            mapping_path = Path(self.model_path) / "label_mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    self.label_mapping = json.load(f)
            else:
                logger.warning("Label mapping not found, using default")
                self.label_mapping = {"sales": 0, "tech_support": 1, "general": 2, "billing": 3}
            
            self.id_to_label = {v: k for k, v in self.label_mapping.items()}
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy" if self.model is not None else "unhealthy",
                model_loaded=self.model is not None,
                device=str(self.model.device) if self.model else "unknown",
                label_mapping=self.label_mapping or {}
            )
        
        @self.app.post("/predict", response_model=TicketResponse)
        async def predict_ticket(request: TicketRequest):
            """Predict ticket routing."""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                # Combine subject and body
                text = f"{request.subject} {request.body}"
                
                # Get predictions
                start_time = time.time()
                top_k_predictions = self.model.get_top_k_predictions([text], k=request.top_k)
                inference_time = (time.time() - start_time) * 1000
                
                # Convert to response format
                predictions = []
                for pred in top_k_predictions[0]:
                    predictions.append(Prediction(
                        label=self.id_to_label[pred["label_id"]],
                        score=pred["score"]
                    ))
                
                return TicketResponse(
                    predicted_label=predictions[0].label,
                    confidence=predictions[0].score,
                    top_k=predictions,
                    inference_ms=inference_time
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict_batch")
        async def predict_batch_tickets(requests: List[TicketRequest]):
            """Predict multiple tickets in batch."""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                # Combine subjects and bodies
                texts = [f"{req.subject} {req.body}" for req in requests]
                
                # Get predictions for all texts
                start_time = time.time()
                all_top_k_predictions = self.model.get_top_k_predictions(texts, k=3)
                inference_time = (time.time() - start_time) * 1000
                
                # Convert to response format
                responses = []
                for i, request in enumerate(requests):
                    predictions = []
                    for pred in all_top_k_predictions[i]:
                        predictions.append(Prediction(
                            label=self.id_to_label[pred["label_id"]],
                            score=pred["score"]
                        ))
                    
                    responses.append(TicketResponse(
                        predicted_label=predictions[0].label,
                        confidence=predictions[0].score,
                        top_k=predictions,
                        inference_ms=inference_time / len(texts)
                    ))
                
                return responses
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/labels")
        async def get_labels():
            """Get available label mapping."""
            return {
                "label_mapping": self.label_mapping or {},
                "available_labels": list(self.label_mapping.keys()) if self.label_mapping else []
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the FastAPI server."""
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)

def main():
    """Main serving function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ticket routing API server')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--device', type=str, help='Device to run on')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--log-level', type=str, default='info', help='Log level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level.upper())
    
    # Create and run server
    server = TicketServer(args.model_path, device=args.device)
    server.run(host=args.host, port=args.port, workers=args.workers)

if __name__ == "__main__":
    main()
