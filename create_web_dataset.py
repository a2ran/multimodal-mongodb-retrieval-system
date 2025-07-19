import os
import json
import requests
import random
from pathlib import Path
from PIL import Image
import time
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebDatasetCreator:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # 다양한 카테고리 키워드
        self.categories = {
            "animals": ["cat", "dog", "bird", "elephant", "tiger", "lion", "rabbit", "horse"],
            "food": ["pizza", "sushi", "coffee", "fruit", "vegetables", "cake", "bread", "pasta"],
            "technology": ["computer", "smartphone", "robot", "drone", "laptop", "ai", "vr", "3d-printer"],
            "nature": ["sunset", "mountain", "forest", "ocean", "flower", "tree", "landscape", "waterfall"],
            "architecture": ["building", "bridge", "house", "skyscraper", "church", "museum", "library", "castle"],
            "people": ["family", "children", "student", "chef", "doctor", "artist", "musician", "farmer"],
            "transportation": ["car", "train", "airplane", "bicycle", "bus", "ship", "motorcycle", "subway"],
            "art": ["painting", "sculpture", "graffiti", "pottery", "photography", "drawing", "installation", "craft"],
            "sports": ["football", "basketball", "swimming", "tennis", "running", "yoga", "skiing", "golf"],
            "education": ["school", "library", "classroom", "laboratory", "graduation", "books", "study", "university"]
        }


    def download_from_picsum(self, num_images=100):
        """Lorem Picsum을 사용하여 이미지 다운로드 (무료, API 키 불필요)"""
        dataset = []
        images_per_category = max(1, num_images // len(self.categories))
        
        for category, keywords in self.categories.items():
            for i in range(images_per_category):
                if len(dataset) >= num_images:
                    break
                
                keyword = random.choice(keywords)
                
                try:
                    # Lorem Picsum에서 랜덤 이미지 가져오기
                    image_id = random.randint(1, 1000)
                    image_url = f"https://picsum.photos/id/{image_id}/800/600"
                    
                    response = requests.get(image_url)
                    
                    if response.status_code == 200:
                        # 파일명 생성
                        filename = f"{category}_{len(dataset)+1:03d}.jpg"
                        filepath = self.images_dir / filename
                        
                        # 이미지 저장
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        # 간단한 설명 생성 (실제 캡션은 없으므로)
                        descriptions = {
                            "animals": f"A beautiful photo featuring {keyword}",
                            "food": f"Delicious {keyword} captured in high quality",
                            "technology": f"Modern {keyword} showcasing innovation",
                            "nature": f"Stunning {keyword} in natural setting",
                            "architecture": f"Impressive {keyword} with architectural beauty",
                            "people": f"People engaged in {keyword} activities",
                            "transportation": f"{keyword} as a mode of transportation",
                            "art": f"Artistic representation of {keyword}",
                            "sports": f"Dynamic {keyword} sports action",
                            "education": f"Educational scene involving {keyword}"
                        }
                        
                        # 데이터셋 엔트리 생성
                        entry = {
                            "id": f"{category}_{len(dataset)+1:03d}",
                            "text": descriptions.get(category, f"An image of {keyword}"),
                            "image_path": str(filepath),
                            "category": category,
                            "keyword": keyword,
                            "metadata": {
                                "category": category,
                                "keyword": keyword,
                                "source": "picsum",
                                "image_id": image_id,
                                "width": 800,
                                "height": 600
                            }
                        }
                        
                        dataset.append(entry)
                        logger.info(f"Downloaded: {filename} - {entry['text']}")
                        
                except Exception as e:
                    logger.error(f"Error downloading image for {keyword}: {e}")
                    continue
                
                # 서버 부하 방지를 위한 딜레이
                time.sleep(0.5)
        
        return dataset


    def create_dataset(self, num_images=100):
        """데이터셋 생성 메인 함수"""
        logger.info(f"Lorem Picsum에서 {num_images}개 이미지 다운로드 시작")
        
        dataset = self.download_from_picsum(num_images)
        
        # 데이터셋 JSON 파일로 저장
        dataset_file = self.data_dir / "web_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 웹 데이터셋이 생성되었습니다!")
        logger.info(f"   - 총 {len(dataset)}개의 샘플")
        logger.info(f"   - 카테고리: {len(set(item['category'] for item in dataset))}개")
        logger.info(f"   - 이미지 저장 위치: {self.images_dir}")
        logger.info(f"   - 데이터셋 파일: {dataset_file}")
        
        return dataset_file


def main():
    """메인 실행 함수"""
    creator = WebDatasetCreator()
    
    print("Lorem Picsum을 사용하여 데이터셋을 생성합니다 (API 키 불필요)")
    
    num_images = input("생성할 이미지 수를 입력하세요 [100]: ").strip()
    try:
        num_images = int(num_images) if num_images else 100
    except ValueError:
        num_images = 100
    
    # 데이터셋 생성
    dataset_file = creator.create_dataset(num_images)
    
    print(f"\n✅ 데이터셋 생성 완료: {dataset_file}")
    print("다음 단계: python demo.py 를 실행하여 데모를 시작하세요!")


if __name__ == "__main__":
    main()