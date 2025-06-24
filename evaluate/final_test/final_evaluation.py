import concurrent.futures
import csv
from datetime import datetime
import json
import os
import re
import sys
import threading
import time
import hashlib
from dotenv import load_dotenv

from anthropic import AnthropicVertex
from google.api_core import exceptions
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import numpy as np
import pandas as pd
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai

# --- Configuration ---
MODEL_LIST = ["gemini-2.0-flash"]
EVALUATOR_MODEL = "gemini-2.5-pro"
HYDE_MODEL = "gemini-2.0-flash"
COMPRESSION_MODEL = "gemini-2.0-flash"

QUESTIONS_FILE = "eval_question_30.jsonl"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
HYDE_CACHE_FILE = "hyde_cache.json"

PROJECT_ID = "turing-berm-q3bf2"
LOCATION = "us-east5"

# --- 초기화 ---    
db = None
clients = {}
hyde_cache = {}

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

def initialize_vertexai():
    """Vertex AI를 초기화합니다."""
    global PROJECT_ID
    print("Vertex AI 초기화 시작...")
    PROJECT_ID = os.getenv("PROJECT_ID")
    if not PROJECT_ID:
        print("오류: .env 파일에 PROJECT_ID를 설정해야 합니다.")
        sys.exit(1)
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print(f"Vertex AI 초기화 완료. (Project: {PROJECT_ID}, Location: {LOCATION})")
    return True

def load_vector_db():
    """FAISS 벡터 DB를 로드합니다."""
    global db
    if db: return True
    print("FAISS 벡터 DB 로딩 시작...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_path = os.path.join(project_root, "embed_faiss")
        if not os.path.exists(local_path):
            print(f"오류: FAISS 벡터 DB 경로를 찾을 수 없습니다: {local_path}")
            return False
        db = FAISS.load_local(local_path, embeddings, allow_dangerous_deserialization=True)
        print("벡터 DB 로딩 완료.")
        return True
    except Exception as e:
        print(f"벡터 DB 로딩 중 오류 발생: {e}")
        return False

def load_hyde_cache():
    """HyDE 캐시를 로드합니다."""
    global hyde_cache
    if os.path.exists(HYDE_CACHE_FILE):
        try:
            with open(HYDE_CACHE_FILE, 'r', encoding='utf-8') as f:
                hyde_cache = json.load(f)
            print(f"HyDE 캐시 로드 완료: {len(hyde_cache)}개 항목")
        except Exception as e:
            print(f"HyDE 캐시 로드 실패: {e}")
            hyde_cache = {}
    else:
        hyde_cache = {}

def save_hyde_cache():
    """HyDE 캐시를 저장합니다."""
    try:
        with open(HYDE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(hyde_cache, f, ensure_ascii=False, indent=2)
        print(f"HyDE 캐시 저장 완료: {len(hyde_cache)}개 항목")
    except Exception as e:
        print(f"HyDE 캐시 저장 실패: {e}")

def get_client(model_name: str):
    """Vertex AI 모델 클라이언트를 로드하고 캐싱합니다."""
    global clients
    if model_name not in clients:
        print(f"INFO: '{model_name}' 모델 클라이언트 로딩 중...")
        if model_name.startswith("claude"):
            if 'anthropic_vertex' not in clients:
                clients['anthropic_vertex'] = AnthropicVertex(project_id=PROJECT_ID, region=LOCATION)
            clients[model_name] = clients['anthropic_vertex']
        else:
            clients[model_name] = GenerativeModel(model_name)
    return clients[model_name]

def count_tokens_approximate(text: str) -> int:
    """텍스트의 대략적인 토큰 수를 계산합니다."""
    # 한국어와 영어 혼재 텍스트에 대한 근사치 계산
    # 영어: 평균 4글자당 1토큰, 한국어: 평균 1.5글자당 1토큰
    korean_chars = len([c for c in text if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3])
    other_chars = len(text) - korean_chars
    return int(korean_chars / 1.5 + other_chars / 4)

def generate_with_retry_and_token_tracking(client, model_name: str, prompt: str, max_retries: int = 3):
    """토큰 사용량을 추적하면서 콘텐츠를 생성합니다."""
    for attempt in range(max_retries):
        try:
            response_text = ""
            input_tokens = count_tokens_approximate(prompt)  # 입력 토큰 근사치
            output_tokens = 0
            
            if model_name.startswith("claude"):
                response = client.messages.create(
                    model=model_name, 
                    max_tokens=2048, 
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
                # Claude API에서 토큰 정보 가져오기
                if hasattr(response, 'usage'):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                else:
                    output_tokens = count_tokens_approximate(response_text)
            else: # Gemini
                response = client.generate_content(prompt)
                if response.candidates and response.candidates[0].content.parts:
                    response_text = response.candidates[0].content.parts[0].text
                    output_tokens = count_tokens_approximate(response_text)
                
                # Gemini의 토큰 정보는 정확하지 않을 수 있으므로 근사치 사용
                if hasattr(response, 'usage_metadata'):
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', input_tokens)
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', output_tokens)
            
            if response_text.strip():
                return {
                    'text': response_text.strip(),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                }
            else:
                print(f"  - 경고: 모델이 빈 응답을 반환했습니다. (시도 {attempt + 1}/{max_retries})")

        except exceptions.ServiceUnavailable as e:
            wait_time = 2 ** (attempt + 1)
            print(f"  - 경고: 서비스 일시 중단(503). {wait_time}초 후 재시도합니다... (시도 {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"  - 오류: 모델 응답 생성 중 예외 발생: {e}")
            break
    
    print(f"  - 오류: 최대 재시도 횟수({max_retries}) 후에도 응답 생성 실패.")
    return {
        'text': f"모델 응답 생성 실패 (모델: {model_name})",
        'input_tokens': input_tokens,
        'output_tokens': 0,
        'total_tokens': input_tokens
    }

def generate_hyde_answer(question: str):
    """HyDE (Hypothetical Document Embeddings)를 위해 가상의 답변을 생성합니다."""
    question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()
    if question_hash in hyde_cache:
        print(f"     - (HyDE) 캐시에서 조회: '{question[:30]}...'")
        return {'text': hyde_cache[question_hash], 'total_tokens': 0}
    
    client = get_client(HYDE_MODEL)
    prompt = f"""다음 질문에 대해 이상적인 답변을 생성해주세요. 이 답변은 사실이 아니어도 괜찮습니다. 
오직 벡터 검색의 품질을 높이기 위한 목적으로만 사용됩니다. 답변은 상세하고 명확하게 작성해주세요.

질문: {question}

답변:"""
    
    result = generate_with_retry_and_token_tracking(client, HYDE_MODEL, prompt)
    if result['text'] and "모델 응답 생성 실패" not in result['text']:
        hyde_cache[question_hash] = result['text']
        print(f"     - (HyDE) 새로 생성: '{question[:30]}...' -> '{result['text'][:50]}...'")
        return result
    else:
        print(f"     - (HyDE) 생성 실패: '{question[:30]}...'. 원본 질문을 대신 사용합니다.")
        return {'text': question, 'total_tokens': 0}

def compress_documents(question: str, documents: list):
    """질문과 관련된 문장만 추출하여 문서를 압축합니다."""
    if not documents:
        return {'text': "", 'total_tokens': 0}
    
    client = get_client(COMPRESSION_MODEL)
    
    all_content = "\n\n".join([f"문서 {i+1}: {doc.page_content}" for i, doc in enumerate(documents)])
    
    prompt = f"""다음 질문에 답하기 위해 필요한 정보만 추출하여 관련 문장들을 정리해주세요.
질문과 직접적으로 관련되지 않은 내용은 제외하고, 핵심 정보만 포함해주세요.

질문: {question}

참고 문서:
{all_content}

질문과 관련된 핵심 정보만 추출하여 정리:"""
    
    result = generate_with_retry_and_token_tracking(client, COMPRESSION_MODEL, prompt)
    if result['text'] and "모델 응답 생성 실패" not in result['text']:
        return result
    else:
        return {'text': all_content, 'total_tokens': 0}

def evaluate_response(question, answer, context, prompt_type, token_info):
    """LLM을 사용하여 생성된 답변을 평가하고 점수를 반환합니다."""
    evaluator_client = get_client(EVALUATOR_MODEL)
    
    if prompt_type == "RAG (Document-based)":
        prompt = f"""당신은 답변의 품질을 엄격하게 평가하는 평가자입니다. 다음 기준에 따라 0점에서 5점 사이로 채점하고, 평가 근거를 한국어로 간략하게 작성해주세요. 반드시 JSON 형식으로만 응답해야 합니다.

[평가 기준]
1. 정확성 (Accuracy): [챗봇 응답]이 사실과 얼마나 일치하는지.
2. 충실도 (Faithfulness): [챗봇 응답]이 **오직 [참고 문서]에 명시된 정보만을 사용**하여 생성되었는지. **문서에 없는 내용, 외부 지식, 또는 과장된 해석이 포함되었다면 0점을 부여하세요.**
3. 관련성 (Relevance): [챗봇 응답]이 [질문]의 의도와 얼마나 관련 있는지.
4. 전문성 (Domain Appropriateness): 방사선 QA 도메인에 적절한 표현과 지식을 사용했는지.
5. 완결성 (Completeness): 답변이 질문에 대해 '참고 문서' 내의 핵심 정보를 빠짐없이 포함하고 있는지.

[평가할 데이터]
- 사용자 질문: "{question}"
- 참고 문서: "{context}"
- 평가할 답변: "{answer}"

[출력 형식]
```json
{{"정확성": {{"score": [0-5], "reason": "평가 근거"}}, "충실도": {{"score": [0-5], "reason": "평가 근거"}}, "관련성": {{"score": [0-5], "reason": "평가 근거"}}, "전문성": {{"score": [0-5], "reason": "평가 근거"}}, "완결성": {{"score": [0-5], "reason": "평가 근거"}}}}
```"""
    else:  # Fallback
        prompt = f"""당신은 답변의 품질을 엄격하게 평가하는 평가자입니다. 다음 기준에 따라 0점에서 5점 사이로 채점하고, 평가 근거를 한국어로 간략하게 작성해주세요. 반드시 JSON 형식으로만 응답해야 합니다.

[평가 기준]
1. 정확성 (Accuracy): [챗봇 응답]이 사실과 얼마나 일치하는지.
2. 충실도 (Faithfulness): 일반 지식 기반 답변이므로 이 항목은 5점으로 고정합니다.
3. 관련성 (Relevance): [챗봇 응답]이 [질문]의 의도와 얼마나 관련 있는지.
4. 전문성 (Domain Appropriateness): 방사선 QA 도메인에 적절한 표현과 지식을 사용했는지.
5. 완결성 (Completeness): 답변이 질문의 모든 측면을 충분히 다루고 있는지.

[평가할 데이터]
- 사용자 질문: "{question}"
- 평가할 답변: "{answer}"

[출력 형식]
```json
{{"정확성": {{"score": [0-5], "reason": "평가 근거"}}, "충실도": {{"score": 5, "reason": "일반 지식 답변"}}, "관련성": {{"score": [0-5], "reason": "평가 근거"}}, "전문성": {{"score": [0-5], "reason": "평가 근거"}}, "완결성": {{"score": [0-5], "reason": "평가 근거"}}}}
```"""

    evaluation_result = generate_with_retry_and_token_tracking(evaluator_client, EVALUATOR_MODEL, prompt)
    
    scores = extract_scores(evaluation_result['text'])
    
    scores['token_info'] = {
        'hyde_tokens': token_info.get('hyde_tokens', 0),
        'condense_tokens': token_info.get('condense_tokens', 0),
        'compression_tokens': token_info.get('compression_tokens', 0),
        'answer_tokens': token_info.get('answer_tokens', 0),
        'evaluation_tokens': evaluation_result['total_tokens'],
        'total_tokens': sum([
            token_info.get('hyde_tokens', 0),
            token_info.get('condense_tokens', 0),
            token_info.get('compression_tokens', 0),
            token_info.get('answer_tokens', 0),
            evaluation_result['total_tokens']
        ])
    }
    
    return scores

def extract_scores(text):
    """응답에서 JSON 점수를 추출합니다."""
    text = re.sub(r'```json\s*|\s*```', '', text)
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        print(f"Error: 응답에서 JSON 객체를 찾을 수 없습니다.\n응답 내용: {text}")
        return {"정확성": {"score": 0, "reason": "JSON 파싱 실패"}, "충실도": {"score": 0, "reason": "JSON 파싱 실패"}, "관련성": {"score": 0, "reason": "JSON 파싱 실패"}, "전문성": {"score": 0, "reason": "JSON 파싱 실패"}, "완결성": {"score": 0, "reason": "JSON 파싱 실패"}}

    json_str = json_match.group(0)
    try:
        result = json.loads(json_str)
        return {
            "정확성": result.get("정확성", {"score": 0, "reason": "점수 없음"}),
            "충실도": result.get("충실도", {"score": 0, "reason": "점수 없음"}),
            "관련성": result.get("관련성", {"score": 0, "reason": "점수 없음"}),
            "전문성": result.get("전문성", {"score": 0, "reason": "점수 없음"}),
            "완결성": result.get("완결성", {"score": 0, "reason": "점수 없음"})
        }
    except json.JSONDecodeError as e:
        print(f"Error: JSON 파싱에 실패했습니다.\n파싱 대상: {json_str}\n에러: {e}")
        return {"정확성": {"score": 0, "reason": "JSON 파싱 실패"}, "충실도": {"score": 0, "reason": "JSON 파싱 실패"}, "관련성": {"score": 0, "reason": "JSON 파싱 실패"}, "전문성": {"score": 0, "reason": "JSON 파싱 실패"}, "완결성": {"score": 0, "reason": "JSON 파싱 실패"}}

def save_results_to_csv(test_run_id, question, model_name, prompt_type, answer, scores, token_info):
    """결과를 CSV 파일에 저장합니다."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"final_evaluation_results_{test_run_id}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'TestRunID', 'Question', 'Model', 'PromptType', 'Answer',
            'Accuracy_Score', 'Accuracy_Reason', 'Faithfulness_Score', 'Faithfulness_Reason',
            'Relevance_Score', 'Relevance_Reason', 'Domain_Score', 'Domain_Reason',
            'Completeness_Score', 'Completeness_Reason',
            'HyDE_Tokens', 'Condense_Tokens', 'Compression_Tokens', 'Answer_Tokens', 
            'Evaluation_Tokens', 'Total_Tokens', 'Timestamp'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'TestRunID': test_run_id,
            'Question': question,
            'Model': model_name,
            'PromptType': prompt_type,
            'Answer': answer,
            'Accuracy_Score': scores.get('정확성', {}).get('score', 0),
            'Accuracy_Reason': scores.get('정확성', {}).get('reason', ''),
            'Faithfulness_Score': scores.get('충실도', {}).get('score', 0),
            'Faithfulness_Reason': scores.get('충실도', {}).get('reason', ''),
            'Relevance_Score': scores.get('관련성', {}).get('score', 0),
            'Relevance_Reason': scores.get('관련성', {}).get('reason', ''),
            'Domain_Score': scores.get('전문성', {}).get('score', 0),
            'Domain_Reason': scores.get('전문성', {}).get('reason', ''),
            'Completeness_Score': scores.get('완결성', {}).get('score', 0),
            'Completeness_Reason': scores.get('완결성', {}).get('reason', ''),
            'HyDE_Tokens': token_info.get('hyde_tokens', 0),
            'Condense_Tokens': token_info.get('condense_tokens', 0),
            'Compression_Tokens': token_info.get('compression_tokens', 0),
            'Answer_Tokens': token_info.get('answer_tokens', 0),
            'Evaluation_Tokens': token_info.get('evaluation_tokens', 0),
            'Total_Tokens': token_info.get('total_tokens', 0),
            'Timestamp': datetime.now().isoformat()
        })
    
    return filepath

def load_questions_from_jsonl(file_path, num_questions=30):
    """JSONL 파일에서 질문을 로드합니다."""
    questions = []
    # 절대 경로로 변환
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > num_questions:
                    break
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        questions.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON 파싱 실패 (라인 {line_num}): {e}")
        print(f"질문 로드 완료: {len(questions)}개")
        return questions
    except FileNotFoundError:
        print(f"Error: 질문 파일을 찾을 수 없습니다: {file_path}")
        return []

def print_summary(results):
    """결과 요약을 출력합니다."""
    if not results:
        print("출력할 결과가 없습니다.")
        return
    
    print("\n" + "="*80)
    print("📊 최종 평가 결과 요약")
    print("="*80)
    
    model_stats = {}
    total_tokens = 0
    
    for result in results:
        model = result['model']
        if model not in model_stats:
            model_stats[model] = {
                'count': 0,
                'accuracy': [],
                'faithfulness': [],
                'relevance': [],
                'domain': [],
                'completeness': [],
                'total_tokens': 0
            }
        
        model_stats[model]['count'] += 1
        model_stats[model]['accuracy'].append(result['scores']['정확성']['score'])
        model_stats[model]['faithfulness'].append(result['scores']['충실도']['score'])
        model_stats[model]['relevance'].append(result['scores']['관련성']['score'])
        model_stats[model]['domain'].append(result['scores']['전문성']['score'])
        model_stats[model]['completeness'].append(result['scores']['완결성']['score'])
        model_stats[model]['total_tokens'] += result['token_info']['total_tokens']
        total_tokens += result['token_info']['total_tokens']
    
    for model, stats in model_stats.items():
        print(f"\n--- 모델: {model} (총 {stats['count']}개 질문) ---")
        print(f"  - 평균 정확성: {np.mean(stats['accuracy']):.2f} / 5")
        print(f"  - 평균 충실도: {np.mean(stats['faithfulness']):.2f} / 5")
        print(f"  - 평균 관련성: {np.mean(stats['relevance']):.2f} / 5")
        print(f"  - 평균 전문성: {np.mean(stats['domain']):.2f} / 5")
        print(f"  - 평균 완결성: {np.mean(stats['completeness']):.2f} / 5")
        print(f"  - 전체 평균 점수: {np.mean([np.mean(stats['accuracy']), np.mean(stats['faithfulness']), np.mean(stats['relevance']), np.mean(stats['domain']), np.mean(stats['completeness'])]):.2f} / 5")
        print(f"  - 총 토큰 사용량: {stats['total_tokens']:,} 토큰")
        print(f"  - 질문당 평균 토큰: {stats['total_tokens'] // stats['count']:,} 토큰")
    
    print(f"\n--- 전체 통계 ---")
    print(f"  - 총 평가 질문 수: {len(results)}")
    print(f"  - 총 토큰 사용량: {total_tokens:,} 토큰")
    print(f"  - 질문당 평균 토큰: {total_tokens // len(results):,} 토큰")
    
    prompt_types = {}
    for result in results:
        prompt_type = result['prompt_type']
        if prompt_type not in prompt_types:
            prompt_types[prompt_type] = 0
        prompt_types[prompt_type] += 1
    
    print(f"\n--- 프롬프트 타입별 사용 현황 ---")
    for prompt_type, count in prompt_types.items():
        print(f"  - {prompt_type}: {count}개 질문")

def prepare_data_for_evaluation(question: str):
    """질문에 대한 RAG 데이터를 준비합니다."""
    print(f"\n🔍 질문 처리 중: '{question[:50]}...'")
    
    token_info = {
        'hyde_tokens': 0,
        'condense_tokens': 0,
        'compression_tokens': 0,
        'answer_tokens': 0,
        'evaluation_tokens': 0,
        'total_tokens': 0
    }
    
    # 1. HyDE 생성
    print("     - HyDE 생성 중...")
    hyde_result = generate_hyde_answer(question)
    token_info['hyde_tokens'] = hyde_result.get('total_tokens', 0)
    hyde_answer = hyde_result['text']
    
    # 2. 질문 요약
    print("     - 질문 요약 중...")
    condensed_result = generate_with_retry_and_token_tracking(
        get_client(HYDE_MODEL), 
        HYDE_MODEL, 
        f"질문을 간결하게 요약해주세요: {question}"
    )
    token_info['condense_tokens'] = condensed_result['total_tokens']
    condensed_question = condensed_result['text']
    
    # 3. 벡터 검색 (HyDE 기반)
    print("     - 벡터 검색 중...")
    docs = db.similarity_search_with_score(hyde_answer, k=10)
    
    # 4. 유사도 기반 RAG/Fallback 결정
    if docs and docs[0][1] <= 1.0:  # 유사도 임계값 조정 (0.5 -> 1.0)
        prompt_type = "RAG (Document-based)"
        print("     - RAG 모드로 답변 생성...")
        
        # 5. 문서 압축
        print("     - 문서 압축 중...")
        compression_result = compress_documents(question, [doc for doc, score in docs if score < 1.0]) # 임계값 조정
        token_info['compression_tokens'] = compression_result['total_tokens']
        
        context = compression_result['text']
    else:
        prompt_type = "Fallback (General Knowledge)"
        print("     - Fallback 모드로 답변 생성...")
        context = "일반 지식 기반 답변"
    
    return {
        'question': question,
        'context': context,
        'prompt_type': prompt_type,
        'token_info': token_info
    }

def generate_single_answer_task(args):
    """단일 질문에 대한 답변 생성 태스크"""
    question, model_name = args
    
    # 데이터 준비
    data = prepare_data_for_evaluation(question)
    context = data['context']
    prompt_type = data['prompt_type']
    token_info = data['token_info']
    
    # 답변 생성
    client = get_client(model_name)
    
    if prompt_type == "RAG (Document-based)":
        prompt = f"""당신은 주어진 [참고 문서]를 바탕으로 답변을 생성하는 방사선 장비 품질관리(QA) 전문가 AI입니다.

[참고 문서]는 당신이 알고 있는 유일한 정보 소스입니다. **절대 외부 지식이나 당신의 기존 지식을 사용하지 마세요.**

[참고 문서]:
{context}

[사용자 질문]:
{question}

[응답 지침]:
1.  **답변 생성**: [사용자 질문]에 답하기 위해, [참고 문서]의 관련 내용을 종합하고 요약하여 자연스러운 전문가의 설명으로 재구성하세요.
2.  **정보 부족 시**: 만약 [참고 문서]의 내용만으로는 질문에 대한 명확한 답변을 생성하기에 불충분하다면, 답변을 지어내지 말고 "제공된 문서를 바탕으로 답변해 드리겠습니다."라고 서두를 시작하며 문서에서 찾은 가장 관련성 높은 정보만으로 설명하세요.
3.  **완전한 정보 부재 시**: 만약 [참고 문서]에 질문과 관련된 내용이 전혀 없다면, "제공된 문서에서는 해당 질문에 대한 정보를 찾을 수 없습니다." 라고만 답변하세요.
4.  **스타일**: 답변은 명확하고 실무적인 전문가의 어조를 유지하세요.

[답변]:"""
    else:
        prompt = f"""당신은 다양한 주제에 대해 답변할 수 있는 유능한 AI 어시스턴트입니다.

사용자 질문: {question}

응답 지침: 당신의 일반 지식을 바탕으로 답변해주세요. 답변 시작 시, "제가 가진 문서에서는 관련 정보를 찾지 못했지만, 일반적인 지식에 따르면" 이라고 명시하세요.

답변:"""
    
    answer_result = generate_with_retry_and_token_tracking(client, model_name, prompt)
    token_info['answer_tokens'] = answer_result['total_tokens']
    
    return {
        'question': question,
        'model': model_name,
        'answer': answer_result['text'],
        'context': context,
        'prompt_type': prompt_type,
        'token_info': token_info
    }

def evaluate_question_across_models(args):
    """하나의 질문에 대해 모든 모델을 평가하는 태스크"""
    question, model_name = args
    
    # 답변 생성
    answer_data = generate_single_answer_task((question, model_name))
    
    # 답변 평가
    scores_with_tokens = evaluate_response(
        answer_data['question'],
        answer_data['answer'],
        answer_data['context'],
        answer_data['prompt_type'],
        answer_data['token_info']
    )
    
    # 최종 토큰 정보를 scores_with_tokens에서 분리
    final_token_info = scores_with_tokens.pop('token_info')
    
    return {
        'question': answer_data['question'],
        'model': answer_data['model'],
        'answer': answer_data['answer'],
        'context': answer_data['context'],
        'prompt_type': answer_data['prompt_type'],
        'scores': scores_with_tokens,
        'token_info': final_token_info
    }

def main():
    """메인 실행 함수"""
    print("🚀 최종 챗봇 성능 평가 시스템 시작")
    print("="*80)
    
    # 초기화
    if not initialize_vertexai():
        return
    if not load_vector_db():
        return
    
    # HyDE 캐시 로드
    load_hyde_cache()
    
    # 질문 로드
    questions = load_questions_from_jsonl(QUESTIONS_FILE)
    if not questions:
        print("질문을 로드할 수 없습니다.")
        return
    
    # 테스트 실행 ID 생성
    test_run_id = f"FinalTest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"테스트 실행 ID: {test_run_id}")
    
    # 모든 질문-모델 조합 생성
    tasks = [(item['question'], model) for item in questions for model in MODEL_LIST]
    print(f"총 {len(tasks)}개의 평가 태스크 준비 완료")
    
    # 병렬 실행
    results = []
    csv_filepath = None
    
    print(f"\n🔄 병렬 평가 시작...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {executor.submit(evaluate_question_across_models, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="평가 진행률"):
            try:
                result = future.result()
                results.append(result)
                
                # CSV에 즉시 저장
                csv_filepath = save_results_to_csv(
                    test_run_id,
                    result['question'],
                    result['model'],
                    result['prompt_type'],
                    result['answer'],
                    result['scores'],
                    result['token_info']
                )
                
                print(f"\n✅ 완료: {result['model']} - '{result['question'][:30]}...'")
                print(f"   - 프롬프트 타입: {result['prompt_type']}")
                print(f"   - 평균 점수: {np.mean([result['scores']['정확성']['score'], result['scores']['충실도']['score'], result['scores']['관련성']['score'], result['scores']['전문성']['score'], result['scores']['완결성']['score']]):.2f}/5")
                print(f"   - 토큰 사용량: {result['token_info']['total_tokens']:,}")
                
            except Exception as e:
                task = future_to_task[future]
                print(f"\n❌ 오류 발생: {task} - {e}")
    
    # HyDE 캐시 저장
    save_hyde_cache()
    
    # 결과 요약 출력
    print_summary(results)
    
    print(f"\n📁 결과 파일 저장 위치: {csv_filepath}")
    print("🎉 최종 평가 완료!")

if __name__ == "__main__":
    main() 