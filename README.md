# Radiation QA ChatBot

## 1. 프로젝트 설명

**Radiation QA ChatBot**은 방사선 품질관리(QA) 문서에 기반하여 사용자의 질문에 정확하고 신뢰성 있는 답변을 제공하는 **RAG 기반 챗봇 시스템**입니다.  
HyDE 기반 임베딩 방식과 캐싱을 적용해 **빠르고 일관된 응답**을 제공하며, GCP의 Vertex AI를 활용해 실제 의료 QA 환경에서 사용할 수 있는 실용성을 목표로 합니다.

![Chatbot Prototype](prenstation/gif/prototype.gif)

- ✅ **문서 기반 QA**: 업로드된 문서를 벡터화하여 의미 기반 검색
- 🔍 **RAG 방식 + HyDE 적용**: 질문의 의미를 확장 후 벡터 검색 → Gemini 모델로 응답 생성
- 🚀 **응답 속도 개선**: LRU 메모리 캐시 + 파일 캐시로 자주 묻는 질문에 빠르게 응답
- 🌐 **Streamlit UI 제공**: 사용자 친화적인 웹 기반 인터페이스

---

## 2. 프로젝트 구조

**SystemArchitecture**
![Chatbot Architecture](prenstation/image/system_architecture.png)

```
CapstoneProject/
├── embed_chunk/        # 문서 임베딩 생성
│   ├── create_embeddings.py
│   └── requirements.txt
├── evaluate/           # 챗봇 성능 평가
│   ├── final_test/
│   ├── unified_test/
│   ├── prompt_test/
│   └── rag_test/
├── fastapi_server/     # FastAPI 서버: 벡터 검색 및 Gemini 응답
│   ├── embed_faiss/    # 생성된 FAISS 인덱스 저장
│   ├── cache.json      # 캐시 파일
│   ├── main.py         # FastAPI 메인 서버 코드
│   ├── nohup.out
│   └── requirements.txt
├── streamlit/          # Streamlit 데모 인터페이스
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
└── README.md
```

---

## 3. 설치 방법

1. **레포지토리 클론**
```bash
git clone https://github.com/your-org/Radiation-QA-ChatBot.git
cd CapstoneProject
```

2. **가상환경 생성 및 활성화**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

3. **라이브러리 설치**
```bash
pip install -r embed_chunk/requirements.txt
pip install -r fastapi_server/requirements.txt
pip install -r streamlit/requirements.txt
```

4. **FAISS 벡터 DB 생성**
```bash
cd embed_chunk
python create_embeddings.py
```

> HyDE를 활용한 질문 임베딩이 자동 생성됨  
> 결과 파일은 `fastapi_server/embed_faiss/`에 저장됨

---

## 4. 사용 방법

1. **FastAPI 서버 실행**
```bash
cd fastapi_server
uvicorn main:app --reload
```

2. **Streamlit 앱 실행**
```bash
cd streamlit
streamlit run app.py
```

3. **브라우저 접속**
```
http://localhost:8501
```

---

## 5. 기술 스택 및 특징

| 구성 요소 | 설명 |
|-----------|------|
| **HyDE 임베딩** | 질문 자체를 임베딩할 뿐 아니라, 질문에서 가상의 답변을 생성 → 이 답변을 임베딩하여 유사 문서 검색 정확도 향상 |
| **Vector Store (FAISS)** | HuggingFace 임베딩을 기반으로 벡터 DB 구성 |
| **Embeddings** | `distiluse-base-multilingual-cased-v1` 모델 사용 |
| **LangChain** | 검색-생성 파이프라인 구성 프레임워크 |
| **Vertex AI Gemini** | 응답 생성에 사용되는 LLM (GenerativeModel 사용) |
| **캐싱 구조** | LRUCache (in-memory) + JSON 파일 캐시 (persistent) 이중 구조로 응답 속도 최적화 |
| **Cloud Run / GCE** | Docker 기반 Streamlit 앱을 GCP에서 배포 가능 |

---

## 6. 향후 개선 사항 (Next Step)

- 📦 PDF 업로드 기능 추가 → 실시간 문서 QA 가능
- 📊 응답 시간 로깅 및 시각화 → 속도 튜닝 및 품질 평가 자동화
- 🧠 도메인 별 프롬프트 튜닝 → 물리학 QA, 간호 QA 등

## 7. 배포 인프라 구성

![Chatbot Deploy](prenstation/image/deploy.png)

- **GCE VM**: 백엔드 (Fastapi 사용)
- **Cloud Run**: 최종 챗봇 배포 환경 (Streamlit, 무중단, 자동 확장)
- **Cloud Build**: 이미지 자동 빌드 및 저장
- **Artifact Registry**: 챗봇 이미지 저장소