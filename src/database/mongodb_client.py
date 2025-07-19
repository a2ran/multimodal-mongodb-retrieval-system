import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class MongoDBClient:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None
        
    def connect(self) -> bool:
        try:
            self.client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
            self.client.admin.command('ping')
            self.db = self.client[os.getenv('MONGODB_DB_NAME', 'multimodal_rag')]
            logger.info("Successfully connected to MongoDB")
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def get_collection(self, collection_name: str):
        if self.db is None:
            self.connect()
        return self.db[collection_name]
    
    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")