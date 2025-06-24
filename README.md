# Capstone Project

## 1. 프로젝트 설명

(여기에 프로젝트에 대한 간략한 설명을 작성하세요. 어떤 문제를 해결하는지, 주요 기능은 무엇인지 등)

## 2. 프로젝트 구조

```
CapstoneProject/
|-- embed_chunk/      # 데이터 전처리 및 임베딩 생성
|-- evaluate/         # 모델 및 RAG 파이프라인 성능 평가
|-- fastapi_server/   # FastAPI를 이용한 RAG API 서버
|-- streamlit/        # Streamlit을 이용한 데모 UI
|-- .gitignore        # Git 버전 관리 제외 파일 목록
`-- README.md         # 프로젝트 설명 파일
```

## 3. 설치 방법

(프로젝트를 로컬 환경에서 실행하기 위해 필요한 절차를 작성하세요. 예를 들어:)

1.  **Repository 클론**
    ```bash
    git clone (레포지토리 주소)
    cd CapstoneProject
    ```

2.  **가상환경 생성 및 활성화**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```

3.  **필요한 라이브러리 설치**
    (각 디렉터리별로 requirements.txt가 있으므로, 설치 방법을 안내합니다.)
    ```bash
    pip install -r fastapi_server/requirements.txt
    pip install -r streamlit/requirements.txt
    pip install -r embed_chunk/requirements.txt
    ```
4. **(필요시) 데이터 및 모델 준비**
   (FAISS 인덱스 생성 방법이나, 다운로드 받아야 할 모델이 있다면 여기에 명시합니다.)


## 4. 사용 방법

(프로젝트 실행 방법을 안내합니다.)

1.  **FastAPI 서버 실행**
    ```bash
    cd fastapi_server
    uvicorn main:app --reload
    ```

2.  **Streamlit 앱 실행**
    ```bash
    cd streamlit
    streamlit run app.py
    ```

## 5. 추가 정보

(알려야 할 추가적인 정보가 있다면 여기에 작성합니다.) 