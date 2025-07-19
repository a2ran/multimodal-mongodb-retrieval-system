import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image
import numpy as np

from ..database.mongodb_client import MongoDBClient
from ..database.schemas import Document, ContentType
from ..models.embeddings import MultimodalEmbedder

logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self):
        self.db_client = MongoDBClient()
        self.db_client.connect()
        self.collection = self.db_client.get_collection("multimodal_documents")
        self.embedder = MultimodalEmbedder()
        
        # Create indexes for efficient retrieval
        self._create_indexes()
    
    def _create_indexes(self):
        # Create indexes for vector search
        self.collection.create_index("content_type")
        self.collection.create_index("created_at")
        self.collection.create_index([("metadata.category", 1)])
        logger.info("Created database indexes")
    
    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        # Generate text embedding
        text_embedding = self.embedder.embed_text(text)[0].tolist()
        
        document = Document(
            content_type=ContentType.TEXT,
            text_content=text,
            text_embedding=text_embedding,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # _id 필드를 제외하고 MongoDB에 저장
        doc_dict = document.dict(by_alias=True, exclude={"_id"})
        result = self.collection.insert_one(doc_dict)
        logger.info(f"Ingested text document with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def ingest_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate image embedding
        image_embedding = self.embedder.embed_image(image_path)[0].tolist()
        
        document = Document(
            content_type=ContentType.IMAGE,
            image_path=image_path,
            image_embedding=image_embedding,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # _id 필드를 제외하고 MongoDB에 저장
        doc_dict = document.dict(by_alias=True, exclude={"_id"})
        result = self.collection.insert_one(doc_dict)
        logger.info(f"Ingested image document with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def ingest_multimodal(self, text: str, image_path: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate embeddings
        text_embedding = self.embedder.embed_text(text)[0].tolist()
        image_embedding = self.embedder.embed_image(image_path)[0].tolist()
        multimodal_embedding = self.embedder.embed_multimodal(text, image_path)[0].tolist()
        
        document = Document(
            content_type=ContentType.MULTIMODAL,
            text_content=text,
            image_path=image_path,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            multimodal_embedding=multimodal_embedding,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # _id 필드를 제외하고 MongoDB에 저장
        doc_dict = document.dict(by_alias=True, exclude={"_id"})
        result = self.collection.insert_one(doc_dict)
        logger.info(f"Ingested multimodal document with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def batch_ingest_texts(self, texts: List[str], 
                          metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        if metadata_list and len(metadata_list) != len(texts):
            raise ValueError("Length of metadata_list must match length of texts")
        
        # Generate embeddings in batch
        text_embeddings = self.embedder.embed_text(texts)
        
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, text_embeddings)):
            metadata = metadata_list[i] if metadata_list else {}
            document = Document(
                content_type=ContentType.TEXT,
                text_content=text,
                text_embedding=embedding.tolist(),
                metadata=metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            # _id 필드를 제외하고 추가
            doc_dict = document.dict(by_alias=True, exclude={"_id"})
            documents.append(doc_dict)
        
        result = self.collection.insert_many(documents)
        logger.info(f"Batch ingested {len(result.inserted_ids)} text documents")
        return [str(id) for id in result.inserted_ids]
    
    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        result = self.collection.update_one(
            {"_id": document_id},
            {
                "$set": {
                    "metadata": metadata,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
    
    def delete_document(self, document_id: str) -> bool:
        result = self.collection.delete_one({"_id": document_id})
        return result.deleted_count > 0