# --- 서버 실행 (백그라운드) ---
# nohup venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 &
#
# --- 서버 로그 확인 ---
# tail -f nohup.out
#
# --- 서버 종료 ---
# pkill -f "uvicorn main:app"
# 또는
# ps aux | grep main  (PID 확인)
# kill [PID]

import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv
from cachetools import LRUCache
import json
import hashlib

# .env 파일로부터 환경 변수 로드
load_dotenv()

# --- 1. 전역 변수 및 모델 로딩 ---
# 캐시 설정 (메모리 + 파일)
CACHE_FILE = "cache.json"
cache = LRUCache(maxsize=200) # 메모리 캐시 (핫 캐시)

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

# Vertex AI 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# 모델들을 담을 딕셔너리
models = {}

def save_cache_to_disk():
    """메모리의 캐시를 디스크(cache.json)에 저장합니다."""
    print("캐시를 디스크에 저장하는 중...")
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(dict(cache.items()), f, ensure_ascii=False, indent=4)
        print(f"'{CACHE_FILE}'에 캐시 저장 완료.")
    except Exception as e:
        print(f"캐시 저장 중 오류 발생: {e}")

def load_cache_from_disk():
    """디스크(cache.json)에서 캐시를 로드하여 메모리에 채웁니다."""
    if os.path.exists(CACHE_FILE):
        print(f"'{CACHE_FILE}'에서 캐시를 로드하는 중...")
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cache.update(data)
                print(f"{len(cache)}개의 항목을 캐시에 로드했습니다.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"캐시 파일 로드 중 오류 발생: {e}")
    else:
        print("캐시 파일이 존재하지 않습니다. 새로운 캐시로 시작합니다.")

def load_models():
    """애플리케이션 시작 시 모델과 캐시를 로드합니다."""
    load_cache_from_disk()
    print("임베딩 모델 및 벡터 DB 로딩 시작...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
    )
    local_path = "embed_faiss"
    db = FAISS.load_local(local_path, embeddings, allow_dangerous_deserialization=True)

    model = GenerativeModel("gemini-2.0-flash")
    hyde_model = GenerativeModel("gemini-2.0-flash")
    helper_model = GenerativeModel("gemini-2.0-flash")

    models['db'] = db
    models['model'] = model
    models['hyde_model'] = hyde_model
    models['helper_model'] = helper_model
    print("임베딩 모델 및 벡터 DB 로딩 완료.")
    print("-" * 50)

# --- 2. FastAPI 앱 설정 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """FastAPI 앱 시작 시 모델과 캐시를 로드합니다."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)

@app.on_event("shutdown")
def shutdown_event():
    """앱 종료 시 메모리 캐시를 디스크에 저장합니다."""
    save_cache_to_disk()

# --- 3. Pydantic 모델 및 Helper 함수 정의 ---
class ChatRequest(BaseModel):
    query: str
    history: list[dict]

async def translate_to_english(text: str, helper_model: GenerativeModel):
    """LLM을 사용하여 텍스트를 영어로 번역합니다."""
    print("     - (Translate) 검색어 영어로 번역 중...")
    prompt = f"다음 한국어 텍스트를 자연스러운 영어로 번역해주세요. 번역 결과 외의 다른 말은 절대 하지 마세요.\n\n한국어: {text}\n\nEnglish:"
    
    response = await helper_model.generate_content_async(prompt)
    translated_text = response.text.strip()

    if translated_text and "모델 응답 생성 실패" not in translated_text:
        print(f"     - (Translate) 번역 완료:\n{translated_text}")
        return translated_text
    else:
        print("     - (Translate) 번역 실패. 원본 텍스트를 사용합니다.")
        return text

async def generate_hyde_answer(question: str, hyde_model: GenerativeModel):
    """HyDE (Hypothetical Document Embeddings)를 위해 가상의 답변을 생성합니다."""
    print("     - (HyDE) 가상 답변 생성 중...")
    prompt = f"""다음 질문에 대해 이상적인 답변을 생성해주세요. 이 답변은 사실이 아니어도 괜찮습니다.
오직 벡터 검색의 품질을 높이기 위한 목적으로만 사용됩니다. 답변은 상세하고 명확하게 작성해주세요.

질문: {question}

답변:"""
    
    response = await hyde_model.generate_content_async(prompt)
    hyde_text = response.text.strip()
    
    if hyde_text:
        print(f"     - (HyDE) 생성 완료:\n{hyde_text}")
        return hyde_text
    else:
        print("     - (HyDE) 생성 실패. 원본 질문을 사용합니다.")
        return question

# --- 4. API 엔드포인트 정의 ---
@app.post("/search")
async def search(request: ChatRequest):
    """
    사용자 질문에 대한 답변을 스트리밍으로 반환합니다. (응답 기반 컨텍스트 캐싱 적용)
    """
    # 1. 캐시 키를 질문(query)만으로 생성하여 히트율을 높임
    cache_key = hashlib.sha256(request.query.encode()).hexdigest()

    # 2. 캐시 확인
    if cache_key in cache:
        print(f"캐시 히트 (컨텍스트로 활용): '{request.query}'")
        cached_answer = cache[cache_key]
        
        # 캐시된 답변을 재활용하여 새로운 프롬프트 구성 (HyDE 및 벡터 검색 생략)
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.history])
        
        prompt = f"""당신은 방사선 장비 전문가 AI 챗봇입니다.

다음은 사용자의 현재 질문과 가장 관련성이 높은 참고 정보입니다.
이 정보를 최우선으로 활용하여 답변을 생성하세요.

[참고 정보]
{cached_answer}
---

다음은 이전 대화 내용입니다.
[이전 대화]
{history_str}
---

위 정보와 대화 내용을 종합하여, 사용자의 질문에 가장 적절한 답변을 자연스러운 한국어로 생성해주세요.

[사용자 질문]
{request.query}

답변:
"""
        print("--- 캐시 기반 프롬프트 ---")
        print(prompt)
        print("--------------------")

        # 캐시된 컨텍스트를 사용하여 답변 스트리밍 (이 답변은 다시 캐시하지 않음)
        model = models.get('model')
        if not model:
            return {"error": "Model is not loaded yet."}

        async def stream_generator():
            try:
                responses = model.generate_content(
                    contents=[Part.from_text(prompt)],
                    stream=True
                )
                for response in responses:
                    yield response.text
            except Exception as e:
                print(f"스트리밍 중 오류 발생: {e}")
            finally:
                print("캐시 기반 스트리밍 종료.")
        
        return StreamingResponse(stream_generator(), media_type="text/plain")

    print(f"캐시 미스: '{request.query}'")
    
    # 캐시 미스 시, 기존의 전체 파이프라인 실행
    db = models.get('db')
    model = models.get('model')
    hyde_model = models.get('hyde_model')
    helper_model = models.get('helper_model')

    if not all([db, model, hyde_model, helper_model]):
        return {"error": "Models are not loaded yet."}

    hyde_query_korean = await generate_hyde_answer(request.query, hyde_model)
    hyde_query_english = await translate_to_english(hyde_query_korean, helper_model)

    print("     - (Search) 한/영 동시 검색 중...")
    docs_ko = db.similarity_search_with_score(hyde_query_korean, k=3)
    docs_en = db.similarity_search_with_score(hyde_query_english, k=3)
    
    combined_docs = sorted(docs_ko + docs_en, key=lambda item: item[1])
    
    unique_docs = []
    seen_contents = set()
    for doc, score in combined_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)
            if len(unique_docs) >= 5:
                break
    
    context = "\n\n".join(doc.page_content for doc in unique_docs)

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.history])

    prompt = f"""당신은 방사선 장비의 품질관리(QA), 유지보수, 안전 점검에 대한 전문 지식을 갖춘 AI 챗봇입니다.

당신의 역할은:
사용자가 제공한 한국어 및 영어 문서 내용과 이전 대화 흐름을 종합적으로 참고하여,
방사선 장비의 작동, 오류 대응, 유지보수, 안전관리 등에 대해
실무적이고 정확하며 자연스러운 한국어로 답변하는 것입니다.

다음은 이전 대화 내용입니다:
{history_str}

다음은 참고 가능한 문서 정보입니다:
{context}

사용자 질문:
{request.query}

응답 지침:
- 제공된 문서가 영어이더라도, 답변은 반드시 유창한 한국어로 작성해야 합니다.
- 문서에 "나와 있지 않다", "직접적으로 나타나진 않는다", "제공된 문서에는" 등의 표현은 절대 사용하지 마세요.
- 문서에 명확한 내용이 없더라도, 전문가의 입장에서 자연스럽게 지식을 바탕으로 설명하세요.
- 설명은 너무 딱딱하지 않게, 하지만 명확하고 실무적으로 서술형으로 작성하세요.
- 기술 용어는 필요한 경우 명확히 설명하고, 문맥에 맞는 예시를 덧붙이세요.
- 답변 길이는 질문의 난이도에 따라 유연하게 조절하세요.

답변:
"""
    print("--- 생성된 프롬프트 ---")
    print(prompt)
    print("--------------------")

    async def stream_generator():
        response_chunks = []
        try:
            # 스트리밍으로 콘텐츠 생성 요청
            responses = model.generate_content(
                contents=[Part.from_text(prompt)],
                stream=True
            )
            for response in responses:
                response_chunks.append(response.text)
                yield response.text
            
            # 스트리밍 종료 후 전체 답변을 캐시에 저장 (오직 전체 파이프라인을 거친 답변만)
            full_response = "".join(response_chunks)
            cache[cache_key] = full_response
            print(f"응답 캐시 완료: '{request.query}'")

        except Exception as e:
            print(f"스트리밍 중 오류 발생: {e}")
        finally:
            print("스트리밍 종료.")

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/")
def read_root():
    return {"message": "GCE FastAPI Server is running."} 