import numpy as np
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import logging

from ..database.mongodb_client import MongoDBClient
from ..database.schemas import SearchQuery, SearchResult, Document, ContentType
from ..models.embeddings import MultimodalEmbedder

logger = logging.getLogger(__name__)


class MultimodalRetriever:
    def __init__(self):
        self.db_client = MongoDBClient()
        self.db_client.connect()
        self.collection = self.db_client.get_collection("multimodal_documents")
        self.embedder = MultimodalEmbedder()
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        # Generate query embedding based on query type
        if query.query_text and query.query_image_path:
            # Multimodal query
            query_embedding = self.embedder.embed_multimodal(
                query.query_text, query.query_image_path
            )[0]
            embedding_field = "multimodal_embedding"
        elif query.query_text:
            # Text-only query
            query_embedding = self.embedder.embed_text(query.query_text)[0]
            embedding_field = "text_embedding"
        elif query.query_image_path:
            # Image-only query
            query_embedding = self.embedder.embed_image(query.query_image_path)[0]
            embedding_field = "image_embedding"
        else:
            raise ValueError("Either query_text or query_image_path must be provided")
        
        # Build MongoDB query
        mongo_query = {}
        
        # Add content type filter if specified
        if query.content_type:
            mongo_query["content_type"] = query.content_type
        
        # Add metadata filters
        if query.metadata_filter:
            for key, value in query.metadata_filter.items():
                mongo_query[f"metadata.{key}"] = value
        
        # Retrieve documents
        documents = list(self.collection.find(mongo_query))
        
        if not documents:
            return []
        
        # Calculate similarities
        results = []
        for doc in documents:
            doc_obj = Document(**doc)
            
            # Get appropriate embedding based on query type
            if embedding_field == "multimodal_embedding" and doc_obj.multimodal_embedding:
                doc_embedding = np.array(doc_obj.multimodal_embedding)
            elif embedding_field == "text_embedding" and doc_obj.text_embedding:
                doc_embedding = np.array(doc_obj.text_embedding)
            elif embedding_field == "image_embedding" and doc_obj.image_embedding:
                doc_embedding = np.array(doc_obj.image_embedding)
            else:
                continue
            
            # Calculate cosine similarity
            similarity = self.embedder.compute_similarity(
                query_embedding.reshape(1, -1), 
                doc_embedding.reshape(1, -1)
            )[0]
            
            # Calculate distance (1 - similarity for cosine)
            distance = 1 - similarity
            
            # Apply threshold filter if specified
            if query.threshold and similarity < query.threshold:
                continue
            
            results.append(SearchResult(
                document=doc_obj,
                score=float(similarity),
                distance=float(distance)
            ))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        return results[:query.top_k]
    
    def search_by_text(self, text: str, top_k: int = 10, 
                      content_type: Optional[ContentType] = None) -> List[SearchResult]:
        query = SearchQuery(
            query_text=text,
            top_k=top_k,
            content_type=content_type
        )
        return self.search(query)
    
    def search_by_image(self, image_path: str, top_k: int = 10,
                       content_type: Optional[ContentType] = None) -> List[SearchResult]:
        query = SearchQuery(
            query_image_path=image_path,
            top_k=top_k,
            content_type=content_type
        )
        return self.search(query)
    
    def search_multimodal(self, text: str, image_path: str, top_k: int = 10) -> List[SearchResult]:
        query = SearchQuery(
            query_text=text,
            query_image_path=image_path,
            top_k=top_k,
            content_type=ContentType.MULTIMODAL
        )
        return self.search(query)
    
    def hybrid_search(self, text: Optional[str] = None, 
                     image_path: Optional[str] = None,
                     text_weight: float = 0.5,
                     top_k: int = 10) -> List[SearchResult]:
        results_dict = {}
        
        # Text search
        if text:
            text_results = self.search_by_text(text, top_k=top_k*2)
            for result in text_results:
                doc_id = str(result.document.id)
                if doc_id not in results_dict:
                    results_dict[doc_id] = {
                        'document': result.document,
                        'text_score': result.score * text_weight,
                        'image_score': 0
                    }
                else:
                    results_dict[doc_id]['text_score'] = result.score * text_weight
        
        # Image search
        if image_path:
            image_weight = 1 - text_weight
            image_results = self.search_by_image(image_path, top_k=top_k*2)
            for result in image_results:
                doc_id = str(result.document.id)
                if doc_id not in results_dict:
                    results_dict[doc_id] = {
                        'document': result.document,
                        'text_score': 0,
                        'image_score': result.score * image_weight
                    }
                else:
                    results_dict[doc_id]['image_score'] = result.score * image_weight
        
        # Combine scores
        final_results = []
        for doc_id, data in results_dict.items():
            combined_score = data['text_score'] + data['image_score']
            final_results.append(SearchResult(
                document=data['document'],
                score=combined_score,
                distance=1 - combined_score
            ))
        
        # Sort and return top_k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]