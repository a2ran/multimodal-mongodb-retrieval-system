import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import shutil
import logging

from src.database.schemas import SearchQuery, ContentType
from src.utils.data_ingestion import DataIngestion
from src.utils.retrieval import MultimodalRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal MongoDB RAG API", version="1.0.0")

# Initialize services
ingestion_service = DataIngestion()
retrieval_service = MultimodalRetriever()

# Create upload directory
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Multimodal MongoDB RAG API", "status": "active"}


@app.post("/ingest/text")
async def ingest_text(
    text: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    try:
        metadata_dict = eval(metadata) if metadata else {}
        doc_id = ingestion_service.ingest_text(text, metadata_dict)
        return {"document_id": doc_id, "status": "success"}
    except Exception as e:
        logger.error(f"Error ingesting text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/image")
async def ingest_image(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        metadata_dict = eval(metadata) if metadata else {}
        doc_id = ingestion_service.ingest_image(str(file_path), metadata_dict)
        
        return {"document_id": doc_id, "status": "success", "file_path": str(file_path)}
    except Exception as e:
        logger.error(f"Error ingesting image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/multimodal")
async def ingest_multimodal(
    text: str = Form(...),
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        metadata_dict = eval(metadata) if metadata else {}
        doc_id = ingestion_service.ingest_multimodal(text, str(file_path), metadata_dict)
        
        return {"document_id": doc_id, "status": "success"}
    except Exception as e:
        logger.error(f"Error ingesting multimodal content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/text")
async def search_by_text(
    query: str = Form(...),
    top_k: int = Form(10),
    content_type: Optional[str] = Form(None)
):
    try:
        content_type_enum = ContentType(content_type) if content_type else None
        results = retrieval_service.search_by_text(query, top_k, content_type_enum)
        
        return {
            "query": query,
            "results": [
                {
                    "document_id": str(result.document.id),
                    "score": result.score,
                    "content_type": result.document.content_type,
                    "text_content": result.document.text_content,
                    "image_path": result.document.image_path,
                    "metadata": result.document.metadata
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    content_type: Optional[str] = Form(None)
):
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"query_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        content_type_enum = ContentType(content_type) if content_type else None
        results = retrieval_service.search_by_image(str(file_path), top_k, content_type_enum)
        
        return {
            "query_image": str(file_path),
            "results": [
                {
                    "document_id": str(result.document.id),
                    "score": result.score,
                    "content_type": result.document.content_type,
                    "text_content": result.document.text_content,
                    "image_path": result.document.image_path,
                    "metadata": result.document.metadata
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error in image search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/multimodal")
async def search_multimodal(
    text: str = Form(...),
    file: UploadFile = File(...),
    top_k: int = Form(10)
):
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"query_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        results = retrieval_service.search_multimodal(text, str(file_path), top_k)
        
        return {
            "query_text": text,
            "query_image": str(file_path),
            "results": [
                {
                    "document_id": str(result.document.id),
                    "score": result.score,
                    "content_type": result.document.content_type,
                    "text_content": result.document.text_content,
                    "image_path": result.document.image_path,
                    "metadata": result.document.metadata
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error in multimodal search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid")
async def hybrid_search(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text_weight: float = Form(0.5),
    top_k: int = Form(10)
):
    try:
        if not text and not file:
            raise ValueError("At least one of text or image must be provided")
        
        image_path = None
        if file:
            file_path = UPLOAD_DIR / f"query_{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_path = str(file_path)
        
        results = retrieval_service.hybrid_search(text, image_path, text_weight, top_k)
        
        return {
            "query_text": text,
            "query_image": image_path,
            "text_weight": text_weight,
            "results": [
                {
                    "document_id": str(result.document.id),
                    "score": result.score,
                    "content_type": result.document.content_type,
                    "text_content": result.document.text_content,
                    "image_path": result.document.image_path,
                    "metadata": result.document.metadata
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)