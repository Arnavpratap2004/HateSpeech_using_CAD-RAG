"""
CAD-RAG FastAPI Backend
Serves the hate speech detection API for the frontend.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn

# Import the CAD-RAG engine
try:
    from src.rag.engine import (
        analyze_sentence,
        pre_retrieval_analysis,
        determine_sentence_type,
        classify_hate_speech_combined,
        tfidf_vectorizer,
        logistic_regression_model,
        agent_executor,
        llm
    )
    ENGINE_LOADED = True
except Exception as e:
    print(f"Warning: Could not load engine: {e}")
    ENGINE_LOADED = False

app = FastAPI(
    title="CAD-RAG API",
    description="Context-Aware Retrieval Augmented Generation for Hate Speech Detection",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    enable_rag: Optional[bool] = True
    threshold: Optional[float] = 0.5


class AnalyzeResponse(BaseModel):
    sentence: str
    prediction: str
    confidence: float
    category: str
    labels: List[str]
    probabilities: Dict[str, float]
    rag_context: Optional[str] = None
    llm_rationale: Optional[str] = None
    entities: Optional[List[str]] = None
    neologisms: Optional[List[str]] = None
    indicator_matches: Optional[Dict[str, List[str]]] = None
    # New CAD-RAG Final Decision fields
    final_label: Optional[str] = None
    override_pre_analysis: Optional[bool] = None
    final_justification: Optional[str] = None
    pre_label: Optional[str] = None
    pre_confidence: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_available: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and component status."""
    return HealthResponse(
        status="healthy",
        model_loaded=ENGINE_LOADED and tfidf_vectorizer is not None,
        rag_available=ENGINE_LOADED and agent_executor is not None
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text for hate speech using CAD-RAG.
    """
    if not ENGINE_LOADED:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Pre-retrieval analysis
        analysis = pre_retrieval_analysis(request.text)
        
        # Category detection
        category = determine_sentence_type(request.text)
        
        # ML Model classification
        model_result = classify_hate_speech_combined(request.text)
        
        # Parse probabilities
        probabilities = {}
        if model_result.get('probabilities'):
            probabilities = {k: float(v) for k, v in model_result['probabilities'].items()}
        
        # Get max probability for confidence
        confidence = max(probabilities.values()) if probabilities else 0.5
        
        # RAG context (if enabled)
        rag_context = None
        if request.enable_rag and agent_executor:
            try:
                rag_output = agent_executor(analysis['entities'], analysis['neologisms'])
                rag_context = rag_output
            except Exception as e:
                rag_context = f"RAG unavailable: {str(e)}"
        
        # LLM rationale
        llm_rationale = None
        if llm:
            try:
                from src.rag.engine import create_dynamic_prompt
                prompt = create_dynamic_prompt(request.text, {"output": rag_context or ""}, category)
                llm_rationale = llm(prompt)
            except Exception as e:
                llm_rationale = f"LLM unavailable: {str(e)}"
        
        # Get full analysis result from engine (includes final decision)
        full_result = analyze_sentence(request.text)
        
        return AnalyzeResponse(
            sentence=request.text,
            prediction=model_result.get('model_result', 'UNKNOWN'),
            confidence=confidence,
            category=category,
            labels=model_result.get('model_labels', []),
            probabilities=probabilities,
            rag_context=full_result.get('rag_context', rag_context),
            llm_rationale=full_result.get('llm_rationale', llm_rationale),
            entities=analysis.get('entities', []),
            neologisms=analysis.get('neologisms', []),
            indicator_matches=analysis.get('indicator_matches', {}),
            # New CAD-RAG Final Decision fields
            final_label=full_result.get('final_label'),
            override_pre_analysis=full_result.get('override_pre_analysis'),
            final_justification=full_result.get('final_justification'),
            pre_label=full_result.get('pre_label'),
            pre_confidence=full_result.get('pre_confidence')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
