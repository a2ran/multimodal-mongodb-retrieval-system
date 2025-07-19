import os
import sys
import json
import logging
import time
import random
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_ingestion import DataIngestion
from src.utils.retrieval import MultimodalRetriever
from src.database.schemas import ContentType
from create_web_dataset import WebDatasetCreator

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalRAGDemo:
    def __init__(self):
        self.ingestion = None
        self.retriever = None
        self.dataset_file = None
        
    def initialize_services(self):
        """서비스 초기화"""
        logger.info("🚀 멀티모달 RAG 시스템 초기화 중...")
        try:
            self.ingestion = DataIngestion()
            self.retriever = MultimodalRetriever()
            logger.info("✅ 서비스 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"❌ 서비스 초기화 실패: {e}")
            logger.error("MongoDB가 실행 중인지 확인하세요: mongod")
            return False
    
    def prepare_dataset(self):
        """데이터셋 준비"""
        dataset_path = Path("data/web_dataset.json")
        
        if not dataset_path.exists():
            logger.info("📥 웹 데이터셋이 없습니다. 새로 생성합니다...")
            creator = WebDatasetCreator()
            self.dataset_file = creator.create_dataset(num_images=50)
        else:
            logger.info("📁 기존 웹 데이터셋을 사용합니다.")
            self.dataset_file = dataset_path
        
        return self.dataset_file
    
    def load_dataset(self):
        """데이터셋 로드"""
        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            logger.info(f"📊 데이터셋 로드 완료: {len(dataset)}개 항목")
            return dataset
        except Exception as e:
            logger.error(f"❌ 데이터셋 로드 실패: {e}")
            return []
    
    def ingest_dataset(self, dataset, limit=30):
        """데이터셋을 MongoDB에 수집"""
        logger.info(f"📤 {min(len(dataset), limit)}개 문서를 MongoDB에 수집 중...")
        
        successful_ingests = 0
        failed_ingests = 0
        
        for i, item in enumerate(dataset[:limit]):
            try:
                if i % 10 == 0:
                    logger.info(f"진행률: {i}/{min(len(dataset), limit)}")
                
                # 이미지 파일 존재 확인
                if not os.path.exists(item['image_path']):
                    logger.warning(f"이미지 파일이 없습니다: {item['image_path']}")
                    failed_ingests += 1
                    continue
                
                # 멀티모달 문서로 수집
                doc_id = self.ingestion.ingest_multimodal(
                    text=item['text'],
                    image_path=item['image_path'],
                    metadata=item['metadata']
                )
                
                successful_ingests += 1
                
            except Exception as e:
                logger.error(f"문서 수집 실패 ({item['id']}): {e}")
                failed_ingests += 1
                continue
        
        logger.info(f"✅ 데이터 수집 완료: 성공 {successful_ingests}개, 실패 {failed_ingests}개")
        return successful_ingests > 0
    
    def demonstrate_text_search(self):
        """텍스트 검색 시연"""
        logger.info("\n" + "="*60)
        logger.info("🔍 텍스트 검색 시연")
        logger.info("="*60)
        
        search_queries = [
            "beautiful nature landscape",
            "modern technology and innovation",
            "delicious food and cooking",
            "animals in their natural habitat",
            "architectural buildings and structures"
        ]
        
        for query in search_queries:
            logger.info(f"\n🔎 검색어: '{query}'")
            try:
                results = self.retriever.search_by_text(query, top_k=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  {i}. 점수: {result.score:.3f}")
                        logger.info(f"     텍스트: {result.document.text_content[:80]}...")
                        logger.info(f"     카테고리: {result.document.metadata.get('category', 'N/A')}")
                        logger.info(f"     키워드: {result.document.metadata.get('keyword', 'N/A')}")
                else:
                    logger.info("  검색 결과가 없습니다.")
            except Exception as e:
                logger.error(f"  검색 오류: {e}")
            
            time.sleep(1)
    
    def demonstrate_image_search(self, dataset):
        """이미지 검색 시연"""
        logger.info("\n" + "="*60)
        logger.info("🖼️  이미지 검색 시연")
        logger.info("="*60)
        
        # 무작위로 몇 개의 이미지를 선택하여 쿼리로 사용
        import random
        query_items = random.sample(dataset, min(3, len(dataset)))
        
        for item in query_items:
            if not os.path.exists(item['image_path']):
                continue
                
            logger.info(f"\n🖼️  쿼리 이미지: {item['image_path']}")
            logger.info(f"    원본 설명: {item['text'][:60]}...")
            
            try:
                results = self.retriever.search_by_image(item['image_path'], top_k=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  {i}. 점수: {result.score:.3f}")
                        logger.info(f"     텍스트: {result.document.text_content[:80]}...")
                        logger.info(f"     카테고리: {result.document.metadata.get('category', 'N/A')}")
                else:
                    logger.info("  검색 결과가 없습니다.")
            except Exception as e:
                logger.error(f"  검색 오류: {e}")
            
            time.sleep(1)
    
    def demonstrate_multimodal_search(self, dataset):
        """멀티모달 검색 시연"""
        logger.info("\n" + "="*60)
        logger.info("🔀 멀티모달 검색 시연")
        logger.info("="*60)
        
        # 텍스트와 이미지를 조합한 검색
        search_combinations = [
            {
                "text": "beautiful landscape with mountains",
                "image_category": "nature"
            },
            {
                "text": "modern architecture and buildings",
                "image_category": "architecture"
            },
            {
                "text": "animals in wildlife",
                "image_category": "animals"
            }
        ]
        
        for combo in search_combinations:
            # 해당 카테고리의 이미지 찾기
            category_items = [item for item in dataset if item['category'] == combo['image_category']]
            if not category_items:
                continue
            
            query_item = random.choice(category_items)
            if not os.path.exists(query_item['image_path']):
                continue
            
            logger.info(f"\n🔀 멀티모달 검색:")
            logger.info(f"    텍스트: '{combo['text']}'")
            logger.info(f"    이미지: {query_item['image_path']}")
            
            try:
                results = self.retriever.search_multimodal(
                    combo['text'], 
                    query_item['image_path'], 
                    top_k=3
                )
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  {i}. 점수: {result.score:.3f}")
                        logger.info(f"     텍스트: {result.document.text_content[:80]}...")
                        logger.info(f"     카테고리: {result.document.metadata.get('category', 'N/A')}")
                else:
                    logger.info("  검색 결과가 없습니다.")
            except Exception as e:
                logger.error(f"  검색 오류: {e}")
            
            time.sleep(1)
    
    def demonstrate_hybrid_search(self, dataset):
        """하이브리드 검색 시연"""
        logger.info("\n" + "="*60)
        logger.info("⚖️  하이브리드 검색 시연")
        logger.info("="*60)
        
        # 다양한 가중치로 하이브리드 검색 테스트
        test_cases = [
            {
                "text": "modern technology",
                "image_category": "technology",
                "text_weight": 0.7,
                "description": "텍스트 중심 (70%)"
            },
            {
                "text": "beautiful nature",
                "image_category": "nature", 
                "text_weight": 0.3,
                "description": "이미지 중심 (30%)"
            },
            {
                "text": "architectural design",
                "image_category": "architecture",
                "text_weight": 0.5,
                "description": "균형 (50%)"
            }
        ]
        
        for case in test_cases:
            # 해당 카테고리의 이미지 찾기
            category_items = [item for item in dataset if item['category'] == case['image_category']]
            if not category_items:
                continue
            
            query_item = random.choice(category_items)
            if not os.path.exists(query_item['image_path']):
                continue
            
            logger.info(f"\n⚖️  하이브리드 검색 ({case['description']}):")
            logger.info(f"    텍스트: '{case['text']}'")
            logger.info(f"    이미지: {query_item['image_path']}")
            logger.info(f"    텍스트 가중치: {case['text_weight']}")
            
            try:
                results = self.retriever.hybrid_search(
                    text=case['text'],
                    image_path=query_item['image_path'],
                    text_weight=case['text_weight'],
                    top_k=3
                )
                
                if results:
                    for i, result in enumerate(results, 1):
                        logger.info(f"  {i}. 점수: {result.score:.3f}")
                        logger.info(f"     텍스트: {result.document.text_content[:80]}...")
                        logger.info(f"     카테고리: {result.document.metadata.get('category', 'N/A')}")
                else:
                    logger.info("  검색 결과가 없습니다.")
            except Exception as e:
                logger.error(f"  검색 오류: {e}")
            
            time.sleep(1)
    
    def run_demo(self):
        """전체 데모 실행"""
        logger.info("🎬 멀티모달 MongoDB RAG 시스템 데모 시작")
        logger.info("="*60)
        
        # 1. 서비스 초기화
        if not self.initialize_services():
            return False
        
        # 2. 데이터셋 준비
        dataset_file = self.prepare_dataset()
        if not dataset_file:
            logger.error("❌ 데이터셋 준비 실패")
            return False
        
        # 3. 데이터셋 로드
        dataset = self.load_dataset()
        if not dataset:
            logger.error("❌ 데이터셋 로드 실패")
            return False
        
        # 4. 데이터 수집
        if not self.ingest_dataset(dataset, limit=30):
            logger.error("❌ 데이터 수집 실패")
            return False
        
        # 5. 검색 시연
        try:
            self.demonstrate_text_search()
            self.demonstrate_image_search(dataset)
            self.demonstrate_multimodal_search(dataset)
            self.demonstrate_hybrid_search(dataset)
            
            logger.info("\n" + "="*60)
            logger.info("🎉 데모 완료!")
            logger.info("="*60)
            logger.info("API 서버를 시작하려면 다음 명령을 실행하세요:")
            logger.info("python -m uvicorn src.api.main:app --reload")
            logger.info("그 후 http://localhost:8000/docs 에서 API를 테스트할 수 있습니다.")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 데모 실행 중 오류: {e}")
            return False


def main():
    """메인 함수"""
    demo = MultimodalRAGDemo()
    
    try:
        success = demo.run_demo()
        if success:
            print("\n✅ 데모가 성공적으로 완료되었습니다!")
        else:
            print("\n❌ 데모 실행 중 오류가 발생했습니다.")
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 데모가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")


if __name__ == "__main__":
    main()