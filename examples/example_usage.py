import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_ingestion import DataIngestion
from src.utils.retrieval import MultimodalRetriever
from src.database.schemas import ContentType
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize services
    logger.info("Initializing services...")
    ingestion = DataIngestion()
    retriever = MultimodalRetriever()
    
    # Example 1: Ingest text documents
    logger.info("\n=== Example 1: Ingesting Text Documents ===")
    text_samples = [
        {
            "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "metadata": {"category": "landmark", "location": "Paris"}
        },
        {
            "text": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": {"category": "programming", "language": "Python"}
        },
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": {"category": "technology", "field": "AI"}
        }
    ]
    
    text_ids = []
    for sample in text_samples:
        doc_id = ingestion.ingest_text(sample["text"], sample["metadata"])
        text_ids.append(doc_id)
        logger.info(f"Ingested text document: {doc_id}")
    
    # Example 2: Search by text
    logger.info("\n=== Example 2: Text Search ===")
    search_query = "artificial intelligence and programming"
    results = retriever.search_by_text(search_query, top_k=3)
    
    logger.info(f"Search query: '{search_query}'")
    for i, result in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"  Score: {result.score:.4f}")
        logger.info(f"  Content: {result.document.text_content[:100]}...")
        logger.info(f"  Metadata: {result.document.metadata}")
    
    # Example 3: Multimodal ingestion (text + image)
    logger.info("\n=== Example 3: Multimodal Document Ingestion ===")
    # Note: You need to have sample images in the data/images directory
    sample_image_path = "data/images/sample_architecture.jpg"
    
    if os.path.exists(sample_image_path):
        multimodal_text = "Modern architecture featuring glass and steel construction with sustainable design elements."
        multimodal_metadata = {
            "category": "architecture",
            "style": "modern",
            "sustainability": "high"
        }
        
        multimodal_id = ingestion.ingest_multimodal(
            multimodal_text, 
            sample_image_path, 
            multimodal_metadata
        )
        logger.info(f"Ingested multimodal document: {multimodal_id}")
        
        # Search for similar multimodal content
        logger.info("\n=== Example 4: Multimodal Search ===")
        search_text = "sustainable building design"
        results = retriever.search_multimodal(search_text, sample_image_path, top_k=5)
        
        logger.info(f"Multimodal search with text: '{search_text}'")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"  Score: {result.score:.4f}")
            logger.info(f"  Text: {result.document.text_content[:100] if result.document.text_content else 'N/A'}...")
            logger.info(f"  Image: {result.document.image_path}")
            logger.info(f"  Metadata: {result.document.metadata}")
    else:
        logger.warning(f"Sample image not found at {sample_image_path}")
        logger.info("Please add sample images to the data/images directory for multimodal examples")
    
    # Example 5: Batch ingestion
    logger.info("\n=== Example 5: Batch Text Ingestion ===")
    batch_texts = [
        "Deep learning revolutionizes computer vision applications.",
        "Natural language processing enables machines to understand human language.",
        "Reinforcement learning allows agents to learn through interaction with environments."
    ]
    batch_metadata = [
        {"category": "AI", "subfield": "computer vision"},
        {"category": "AI", "subfield": "NLP"},
        {"category": "AI", "subfield": "RL"}
    ]
    
    batch_ids = ingestion.batch_ingest_texts(batch_texts, batch_metadata)
    logger.info(f"Batch ingested {len(batch_ids)} documents")
    
    # Example 6: Filtered search
    logger.info("\n=== Example 6: Filtered Search ===")
    filtered_results = retriever.search_by_text(
        "learning", 
        top_k=5,
        content_type=ContentType.TEXT
    )
    
    logger.info("Search for 'learning' with content_type=TEXT filter:")
    for i, result in enumerate(filtered_results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"  Score: {result.score:.4f}")
        logger.info(f"  Content: {result.document.text_content[:100]}...")
        logger.info(f"  Category: {result.document.metadata.get('category', 'N/A')}")
    
    logger.info("\n=== Example Usage Complete ===")


if __name__ == "__main__":
    main()