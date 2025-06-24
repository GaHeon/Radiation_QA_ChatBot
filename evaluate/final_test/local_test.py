import os
import sys
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part

# --- 0. 프로젝트 루트 경로 설정 ---
# 현재 파일의 위치를 기준으로 프로젝트 루트를 계산하여 sys.path에 추가
# 이렇게 하면 다른 디렉토리의 모듈이나 파일을 쉽게 임포트할 수 있습니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# --- 1. 환경 변수 및 모델 로딩 ---
# .env 파일이 프로젝트 루트에 있다고 가정하고 로드
dotenv_path = os.path.join(project_root, 'fastapi_server', '.env')
load_dotenv(dotenv_path=dotenv_path)

# GCP 프로젝트/리전 설정
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")

# Vertex AI 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# FAISS DB 및 모델 로드
print("임베딩 모델 및 벡터 DB 로딩 시작...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
)
# FAISS DB 경로 설정 (fastapi_server 디렉토리 기준)
db_path = os.path.join(project_root, "fastapi_server", "embed_faiss")
db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# Gemini 모델 로드
model = GenerativeModel("gemini-2.0-flash")
hyde_model = GenerativeModel("gemini-2.0-flash") # HyDE용 모델
helper_model = GenerativeModel("gemini-2.0-flash") # 번역용 모델 추가
print("임베딩 모델 및 벡터 DB 로딩 완료.")
print("-" * 50)


def translate_to_english(text: str):
    """LLM을 사용하여 텍스트를 영어로 번역합니다."""
    print("     - (Translate) 검색어 영어로 번역 중...")
    prompt = f"다음 한국어 텍스트를 자연스러운 영어로 번역해주세요. 번역 결과 외의 다른 말은 절대 하지 마세요.\n\n한국어: {text}\n\nEnglish:"
    
    response = helper_model.generate_content(prompt)
    translated_text = response.text.strip()

    if translated_text and "모델 응답 생성 실패" not in translated_text:
        print(f"     - (Translate) 번역 완료: '{translated_text[:80]}...'")
        return translated_text
    else:
        print("     - (Translate) 번역 실패. 원본 텍스트를 사용합니다.")
        return text

def generate_hyde_answer(question: str):
    """HyDE (Hypothetical Document Embeddings)를 위해 가상의 답변을 생성합니다."""
    print("     - (HyDE) 가상 답변 생성 중...")
    prompt = f"""다음 질문에 대해 이상적인 답변을 생성해주세요. 이 답변은 사실이 아니어도 괜찮습니다.
오직 벡터 검색의 품질을 높이기 위한 목적으로만 사용됩니다. 답변은 상세하고 명확하게 작성해주세요.

질문: {question}

답변:"""
    
    response = hyde_model.generate_content(prompt)
    hyde_text = response.text.strip()
    
    if hyde_text:
        print(f"     - (HyDE) 생성 완료: '{hyde_text[:80]}...'")
        return hyde_text
    else:
        print("     - (HyDE) 생성 실패. 원본 질문을 사용합니다.")
        return question


def get_response(query: str, history: list):
    """
    사용자 질문에 대한 답변과 토큰 수를 반환합니다.
    """
    # 1. HyDE를 통해 가상 답변 생성 (한국어)
    hyde_query_korean = generate_hyde_answer(query)
    
    # 2. 생성된 HyDE 답변을 영어로 번역
    hyde_query_english = translate_to_english(hyde_query_korean)

    # 3. 한글/영어 검색어로 각각 FAISS 벡터 검색 수행
    print("     - (Search) 한/영 동시 검색 중...")
    docs_ko = db.similarity_search_with_score(hyde_query_korean, k=3)
    docs_en = db.similarity_search_with_score(hyde_query_english, k=3)
    
    # 4. 검색 결과 통합 및 중복 제거
    # 점수(낮을수록 좋음) 기준으로 정렬 후, 내용이 중복된 문서 제거
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

    # 5. 이전 대화 기록 통합
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    # 6. 프롬프트 구성 (사용자 원본 질문을 사용)
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
{query}

응답 지침:
- 제공된 문서가 영어이더라도, 답변은 반드시 유창한 한국어로 작성해야 합니다.
- 문서에 "나와 있지 않다", "직접적으로 나타나진 않는다", "제공된 문서에는" 등의 표현은 절대 사용하지 마세요.
- 문서에 명확한 내용이 없더라도, 전문가의 입장에서 자연스럽게 지식을 바탕으로 설명하세요.
- 설명은 너무 딱딱하지 않게, 하지만 명확하고 실무적으로 서술형으로 작성하세요.
- 기술 용어는 필요한 경우 명확히 설명하고, 문맥에 맞는 예시를 덧붙이세요.
- 답변 길이는 질문의 난이도에 따라 유연하게 조절하세요.

답변:
"""

    # 7. 모델 응답 생성 (스트리밍 없이)
    response = model.generate_content(
        contents=[Part.from_text(prompt)]
    )
    
    # 8. 토큰 수 계산
    # 입력 프롬프트의 토큰 수를 계산합니다.
    prompt_tokens = model.count_tokens(contents=[Part.from_text(prompt)])
    # 생성된 답변의 토큰 수를 계산합니다.
    response_tokens = model.count_tokens(response.candidates[0].content)

    return {
        "answer": response.text,
        "prompt_tokens": prompt_tokens.total_tokens,
        "response_tokens": response_tokens.total_tokens
    }

def load_questions_from_jsonl(file_path):
    """JSONL 파일에서 질문 목록을 로드합니다."""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

# --- 5. 일괄 처리 테스트 실행 ---
if __name__ == "__main__":
    # 질문 파일 경로 설정
    questions_file_path = os.path.join(project_root, 'evaluate', 'final_test', 'eval_question_30.jsonl')
    
    if not os.path.exists(questions_file_path):
        print(f"오류: 질문 파일을 찾을 수 없습니다 - {questions_file_path}")
    else:
        questions_data = load_questions_from_jsonl(questions_file_path)
        
        total_prompt_tokens = 0
        total_response_tokens = 0
        
        print(f"총 {len(questions_data)}개의 질문에 대한 평가를 시작합니다...")
        print("-" * 70)

        for i, item in enumerate(questions_data):
            user_query = item['question']
            
            print(f"[{i+1}/{len(questions_data)}] 질문: {user_query}")
            
            result = get_response(user_query, []) # 대화 기록은 비운 상태로 처리
            
            print("\n[챗봇 답변]:")
            print(result["answer"])
            print("-" * 20)
            print(f"입력 토큰: {result['prompt_tokens']}, 출력 토큰: {result['response_tokens']}")
            print("-" * 70)
            
            total_prompt_tokens += result['prompt_tokens']
            total_response_tokens += result['response_tokens']
            
        # --- 최종 결과 요약 ---
        num_questions = len(questions_data)
        avg_prompt_tokens = total_prompt_tokens / num_questions
        avg_response_tokens = total_response_tokens / num_questions
        avg_total_tokens = (total_prompt_tokens + total_response_tokens) / num_questions

        print("\n" + "="*30)
        print("     최종 토큰 사용량 요약")
        print("="*30)
        print(f"총 질문 수: {num_questions}개")
        print(f"평균 입력 토큰: {avg_prompt_tokens:.2f}")
        print(f"평균 출력 토큰: {avg_response_tokens:.2f}")
        print(f"평균 총 토큰: {avg_total_tokens:.2f}")
        print("="*30) 