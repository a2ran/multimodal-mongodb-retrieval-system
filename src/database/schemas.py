from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class Document(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    content_type: ContentType
    text_content: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    multimodal_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchQuery(BaseModel):
    query_text: Optional[str] = None
    query_image_path: Optional[str] = None
    query_image_url: Optional[str] = None
    content_type: Optional[ContentType] = None
    top_k: int = 10
    threshold: Optional[float] = None
    metadata_filter: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    document: Document
    score: float
    distance: float