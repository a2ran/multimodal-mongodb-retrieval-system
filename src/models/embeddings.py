import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CLIP model for image and multimodal embeddings
        clip_model_name = os.getenv('CLIP_MODEL_NAME', 'openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Sentence transformer for text embeddings
        text_model_name = os.getenv('TEXT_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
        self.text_model = SentenceTransformer(text_model_name)
        
        logger.info(f"Initialized models on {self.device}")
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.text_model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def embed_image(self, images: Union[Image.Image, List[Image.Image], str, List[str]]) -> np.ndarray:
        if isinstance(images, (str, Image.Image)):
            images = [images]
        
        # Load images if paths are provided
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        # Process images with CLIP
        inputs = self.clip_processor(images=loaded_images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_embeddings = image_features.cpu().numpy()
        
        # Normalize embeddings
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        
        return image_embeddings
    
    def embed_multimodal(self, texts: Union[str, List[str]], 
                        images: Union[Image.Image, List[Image.Image], str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(images, (str, Image.Image)):
            images = [images]
        
        # Load images if paths are provided
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert('RGB'))
            else:
                loaded_images.append(img)
        
        # Process with CLIP
        inputs = self.clip_processor(text=texts, images=loaded_images, 
                                   return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Combine image and text features
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Average pooling of image and text embeddings
            multimodal_embeds = (image_embeds + text_embeds) / 2
            multimodal_embeds = multimodal_embeds.cpu().numpy()
        
        # Normalize
        multimodal_embeds = multimodal_embeds / np.linalg.norm(multimodal_embeds, axis=1, keepdims=True)
        
        return multimodal_embeds
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        # Cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(doc_norms, query_norm.T).flatten()
        return similarities