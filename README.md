# Multimodal MongoDB RAG System

텍스트와 이미지를 함께 처리할 수 있는 멀티모달 RAG(Retrieval-Augmented Generation) 시스템입니다. MongoDB를 벡터 데이터베이스로 사용하여 텍스트, 이미지, 그리고 텍스트-이미지 멀티모달 데이터를 저장하고 검색할 수 있습니다.

## 주요 기능

- 🔍 **멀티모달 검색**: 텍스트, 이미지, 또는 둘 다를 사용한 MongoDB 기반 검색시스템
- 🤖 **오픈소스 임베딩**: 오픈소스 CLIP과 Sentence Transformers를 활용한 임베딩 구조
- 🔄 **하이브리드 검색**: 텍스트와 이미지 검색 결과를 가중치 기반으로 결합

## 시스템 아키텍처

```
multimodal-mongodb-retrieval-system/
├── src/
│   ├── api/              # FastAPI 엔드포인트
│   ├── database/         # MongoDB 연결 및 스키마
│   ├── models/          # 임베딩 모델 (CLIP, Sentence Transformers)
│   └── utils/           # 데이터 수집 및 검색 유틸리티
├── data/                # 데이터 저장 디렉토리
├── examples/            # 사용 예제
└── tests/              # 테스트 코드
```

## 설치 방법

### 1. 필수 요구사항

- Python 3.8+
- MongoDB 4.4+

### 2. 프로젝트 클론

```bash
git clone https://github.com/yourusername/multimodal-mongodb-retrieval-system.git
cd multimodal-mongodb-retrieval-system
```

### 3. 가상환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 4. 의존성 설치

```bash
pip install -r requirements.txt
```

### 5. 환경 변수 설정

기본 설정 파일(`.env`)이 이미 포함되어 있습니다. 필요시 편집하여 MongoDB 연결 정보를 수정할 수 있습니다:

```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=multimodal_rag
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
TEXT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## 사용 방법

### 1. MongoDB 설치 및 시작

#### Docker 사용:
```bash
# MongoDB 컨테이너 실행
docker run -d --name mongodb -p 27017:27017 mongo:7.0

# 상태 확인
docker ps
```

### 2. 데모 실행


```bash
# 1. 웹 데이터셋 생성 (100개 이미지)
python create_web_dataset.py

# 2. 데모 실행
python demo.py
```

데모는 다음과 같은 기능을 시연합니다:
- 웹에서 실제 이미지 다운로드 및 데이터베이스 수집
- 텍스트 검색 (5가지 쿼리)
- 이미지 검색 (유사 이미지 찾기)
- 멀티모달 검색 (텍스트 + 이미지)
- 하이브리드 검색 (가중치 조절)

### 3. API 서버 실행

```bash
# 프로젝트 루트 디렉토리에서 실행
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API 문서는 http://localhost:8000/docs 에서 확인할 수 있습니다.

### 4. 기본 사용 예제

#### 텍스트 데이터 수집

```python
from src.utils.data_ingestion import DataIngestion

ingestion = DataIngestion()

# 텍스트 문서 추가
doc_id = ingestion.ingest_text(
    text="인공지능은 인간의 지능을 모방하는 기술입니다.",
    metadata={"category": "AI", "language": "ko"}
)
```

#### 이미지 데이터 수집

```python
# 이미지 문서 추가
doc_id = ingestion.ingest_image(
    image_path="path/to/image.jpg",
    metadata={"category": "architecture", "style": "modern"}
)
```

#### 멀티모달 데이터 수집

```python
# 텍스트와 이미지가 함께 있는 문서 추가
doc_id = ingestion.ingest_multimodal(
    text="현대적인 건축물의 유리와 철골 구조",
    image_path="path/to/building.jpg",
    metadata={"category": "architecture"}
)
```

#### 검색 수행

```python
from src.utils.retrieval import MultimodalRetriever

retriever = MultimodalRetriever()

# 텍스트로 검색
results = retriever.search_by_text("인공지능 기술", top_k=5)

# 이미지로 검색
results = retriever.search_by_image("query_image.jpg", top_k=5)

# 멀티모달 검색 (텍스트 + 이미지)
results = retriever.search_multimodal(
    text="모던 건축",
    image_path="query_building.jpg",
    top_k=5
)

# 하이브리드 검색 (가중치 조절 가능)
results = retriever.hybrid_search(
    text="건축 디자인",
    image_path="query_image.jpg",
    text_weight=0.7,  # 텍스트 가중치 70%
    top_k=10
)
```

## API 엔드포인트

### 데이터 수집 (Ingestion)

#### POST `/ingest/text`
텍스트 문서를 수집합니다.

```bash
curl -X POST "http://localhost:8000/ingest/text" \
  -F "text=인공지능은 미래 기술입니다" \
  -F 'metadata={"category": "tech"}'
```

#### POST `/ingest/image`
이미지를 수집합니다.

```bash
curl -X POST "http://localhost:8000/ingest/image" \
  -F "file=@image.jpg" \
  -F 'metadata={"category": "photo"}'
```

#### POST `/ingest/multimodal`
텍스트와 이미지를 함께 수집합니다.

```bash
curl -X POST "http://localhost:8000/ingest/multimodal" \
  -F "text=아름다운 풍경 사진" \
  -F "file=@landscape.jpg" \
  -F 'metadata={"category": "nature"}'
```

### 검색 (Search)

#### POST `/search/text`
텍스트 쿼리로 검색합니다.

```bash
curl -X POST "http://localhost:8000/search/text" \
  -F "query=인공지능" \
  -F "top_k=10"
```

#### POST `/search/image`
이미지로 검색합니다.

```bash
curl -X POST "http://localhost:8000/search/image" \
  -F "file=@query_image.jpg" \
  -F "top_k=10"
```

#### POST `/search/multimodal`
텍스트와 이미지를 모두 사용하여 검색합니다.

```bash
curl -X POST "http://localhost:8000/search/multimodal" \
  -F "text=현대 건축물" \
  -F "file=@building.jpg" \
  -F "top_k=10"
```

#### POST `/search/hybrid`
가중치 기반 하이브리드 검색을 수행합니다.

```bash
curl -X POST "http://localhost:8000/search/hybrid" \
  -F "text=건축 디자인" \
  -F "file=@architecture.jpg" \
  -F "text_weight=0.6" \
  -F "top_k=10"
```

## 고급 기능

### 1. 메타데이터 필터링

검색 시 메타데이터를 기반으로 필터링할 수 있습니다:

```python
from src.database.schemas import SearchQuery

query = SearchQuery(
    query_text="기계학습",
    metadata_filter={"category": "AI", "language": "ko"},
    top_k=10
)
results = retriever.search(query)
```

### 2. 배치 처리

여러 문서를 한 번에 처리할 수 있습니다:

```python
texts = ["문서1", "문서2", "문서3"]
metadata_list = [
    {"category": "cat1"},
    {"category": "cat2"},
    {"category": "cat3"}
]

doc_ids = ingestion.batch_ingest_texts(texts, metadata_list)
```

### 3. 임베딩 모델 커스터마이징

`.env` 파일에서 다른 모델을 지정할 수 있습니다:

```env
CLIP_MODEL_NAME=openai/clip-vit-large-patch14
TEXT_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
```