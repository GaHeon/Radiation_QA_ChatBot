import os
import sys
import csv
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from vertexai.generative_models import GenerativeModel
from anthropic import AnthropicVertex
import vertexai

# --- 설정 ---
# 테스트할 Vertex AI 모델 ID 목록
MODEL_LIST = [
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "claude-3-5-sonnet-v2",
]
# 평가에 사용할 모델
EVALUATOR_MODEL = "gemini-2.0-flash"
# 결과 저장 파일 경로
RESULTS_CSV_PATH = 'evaluate/prompt_test/test_results.csv'

# --- 초기화 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()
db = None
clients = {} # 모델/클라이언트 객체를 캐싱하기 위한 딕셔너리
PROJECT_ID = None
LOCATION = "us-east5" # Claude 3.5 Sonnet 사용 가능 리전으로 고정

def initialize_vertexai():
    """Vertex AI를 초기화합니다."""
    global PROJECT_ID, LOCATION
    print("환경 변수 및 Vertex AI 초기화 시작...")
    PROJECT_ID = os.getenv("PROJECT_ID")
    if not PROJECT_ID:
        print("오류: .env 파일에 PROJECT_ID를 설정해야 합니다.")
        return False
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

def get_client(model_name: str):
    """Vertex AI 모델 클라이언트를 로드하고 캐싱합니다."""
    global clients, PROJECT_ID, LOCATION
    if model_name not in clients:
        print(f"INFO: '{model_name}' 모델/클라이언트 로딩 중...")
        if model_name.startswith("claude"):
            # Anthropic 클라이언트는 한번만 생성하여 재사용합니다.
            if 'anthropic_vertex' not in clients:
                # 디버깅: Claude 클라이언트 초기화 직전의 값들을 확인합니다.
                print(f"DEBUG: Initializing AnthropicVertex -> Project ID: '{PROJECT_ID}', Location: '{LOCATION}'")
                if not PROJECT_ID or not LOCATION:
                    print("CRITICAL: Vertex AI가 초기화되지 않았습니다. Claude 클라이언트를 생성할 수 없습니다.")
                    clients[model_name] = None
                    return None
                clients['anthropic_vertex'] = AnthropicVertex(project_id=PROJECT_ID, region=LOCATION)
            # Claude 모델의 경우, 실제 모델 이름은 호출 시 전달되므로 클라이언트 자체를 반환합니다.
            # 혼동을 피하기 위해 모델 이름으로도 클라이언트를 저장합니다.
            clients[model_name] = clients['anthropic_vertex']
        else: # Google 모델
            clients[model_name] = GenerativeModel(model_name)
    return clients[model_name]

def extract_json_from_string(s: str) -> dict:
    """문자열에서 JSON 블록을 추출하여 파싱합니다."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            print("  - 경고: 평가자 모델의 JSON 응답 파싱 실패")
            return {}
    return {}

def evaluate_response(question, answer, context, prompt_type):
    """LLM을 사용하여 생성된 답변을 평가하고 점수를 반환합니다."""
    evaluator_client = get_client(EVALUATOR_MODEL)

    if prompt_type == "RAG (Document-based)":
        prompt = f"""당신은 답변의 품질을 엄격하게 평가하는 평가자입니다. 다음 기준에 따라 0점에서 5점 사이로 채점하고, 평가 근거를 한국어로 간략하게 작성해주세요. 반드시 JSON 형식으로만 응답해야 합니다.
        [평가 기준]
        1. 충실도 (Faithfulness): 답변이 주어진 '참고 문서'의 내용에만 근거하여 생성되었는가? 외부 지식을 사용하지 않았는가?
        2. 정확성 (Accuracy): 답변이 질문에 대해 사실적으로 정확한가?
        3. 관련성 (Relevance): 답변이 사용자의 질문 의도와 얼마나 관련이 있는가?
        4. 완결성 (Completeness): 답변이 질문에 대해 '참고 문서' 내의 핵심 정보를 빠짐없이 포함하고 있는가?
        [평가할 데이터]
        - 사용자 질문: "{question}"
        - 참고 문서: "{context}"
        - 평가할 답변: "{answer}"
        [출력 형식]
        ```json
        {{"충실도": {{"score": [0-5], "reason": "평가 근거"}}, "정확성": {{"score": [0-5], "reason": "평가 근거"}}, "관련성": {{"score": [0-5], "reason": "평가 근거"}}, "완결성": {{"score": [0-5], "reason": "평가 근거"}}}}
        ```"""
    else:  # Fallback
        prompt = f"""당신은 답변의 품질을 엄격하게 평가하는 평가자입니다. 다음 기준에 따라 0점에서 5점 사이로 채점하고, 평가 근거를 한국어로 간략하게 작성해주세요. 반드시 JSON 형식으로만 응답해야 합니다.
        [평가 기준]
        1. 충실도 (Faithfulness): 일반 지식 기반 답변이므로 이 항목은 5점으로 고정합니다.
        2. 정확성 (Accuracy): 답변이 일반적으로 알려진 사실에 비추어 정확한가?
        3. 관련성 (Relevance): 답변이 사용자의 질문 의도와 얼마나 관련이 있는가?
        4. 완결성 (Completeness): 답변이 질문의 모든 측면을 충분히 다루고 있는가?
        [평가할 데이터]
        - 사용자 질문: "{question}"
        - 평가할 답변: "{answer}"
        [출력 형식]
        ```json
        {{"충실도": {{"score": 5, "reason": "일반 지식 답변"}}, "정확성": {{"score": [0-5], "reason": "평가 근거"}}, "관련성": {{"score": [0-5], "reason": "평가 근거"}}, "완결성": {{"score": [0-5], "reason": "평가 근거"}}}}
        ```"""
    try:
        response_text = ""
        if EVALUATOR_MODEL.startswith("claude"):
            response = evaluator_client.messages.create(
                model=EVALUATOR_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text
        else: # Gemini
            response = evaluator_client.generate_content(prompt)
            response_text = response.text

        return extract_json_from_string(response_text)
    except Exception as e:
        print(f"  - 평가 중 오류 발생: {e}")
        return {}

def generate_answer_for_model(model_name: str, query: str):
    """특정 모델에 대해 답변을 생성합니다."""
    global db
    client = get_client(model_name)
    
    docs = db.similarity_search_with_score(query, k=10)
    
    prompt_type = "Fallback (General Knowledge)"
    context = "N/A"
    
    if docs and docs[0][1] <= 0.5:
        prompt_type = "RAG (Document-based)"
        context = "\n\n".join(f"문서 {i+1} (유사도: {score:.4f}):\n{doc.page_content}" for i, (doc, score) in enumerate(docs) if score < 0.55)
        
        prompt = f"""당신은 제공된 문서를 기반으로 답변하는 전문 AI 어시스턴트입니다.
        다음은 참고 가능한 문서 정보입니다:\n---\n{context}\n---\n사용자 질문: {query}\n
        응답 지침: 반드시 위에 제공된 '참고 가능한 문서 정보'만을 사용하여 답변해야 합니다. 사전 지식을 사용해서는 안 됩니다.
        답변:"""
    else:
        prompt = f"""당신은 다양한 주제에 대해 답변할 수 있는 유능한 AI 어시스턴트입니다.
        사용자 질문: {query}\n
        응답 지침: 당신의 일반 지식을 바탕으로 답변해주세요. 답변 시작 시, "제가 가진 문서에서는 관련 정보를 찾지 못했지만, 일반적인 지식에 따르면" 이라고 명시하세요.
        답변:"""
        
    try:
        answer = ""
        if model_name.startswith("claude"):
            message = client.messages.create(
                model=model_name,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            answer = message.content[0].text
        else: # Gemini
            response = client.generate_content(prompt)
            answer = response.text

    except Exception as e:
        answer = f"모델 응답 생성 중 오류 발생: {e}"
        
    return prompt_type, context, answer

def save_results_to_csv(test_run_id, question, model_name, prompt_type, context, answer, scores):
    """테스트 결과를 CSV 파일에 추가합니다."""
    file_exists = os.path.isfile(RESULTS_CSV_PATH)
    with open(RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['TestRunID', 'Timestamp', 'Question', 'Model', 'PromptType', 'Answer', 'Context', 'Score_Faithfulness', 'Reason_Faithfulness', 'Score_Accuracy', 'Reason_Accuracy', 'Score_Relevance', 'Reason_Relevance', 'Score_Completeness', 'Reason_Completeness'])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([
            test_run_id, timestamp, question, model_name, prompt_type, answer, context,
            scores.get("충실도", {}).get("score", ""), scores.get("충실도", {}).get("reason", ""),
            scores.get("정확성", {}).get("score", ""), scores.get("정확성", {}).get("reason", ""),
            scores.get("관련성", {}).get("score", ""), scores.get("관련성", {}).get("reason", ""),
            scores.get("완결성", {}).get("score", ""), scores.get("완결성", {}).get("reason", "")
        ])

def load_questions_from_jsonl(file_path, num_questions=10):
    """JSONL 파일에서 질문을 로드합니다."""
    questions = []
    print(f"질문 파일 로딩 시작: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_questions:
                    break
                data = json.loads(line)
                if 'question' in data:
                    questions.append(data['question'])
                else:
                    print(f"  - 경고: {i+1}번째 라인에 'question' 키가 없습니다.")
        print(f"총 {len(questions)}개의 질문을 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: 질문 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"질문 파일 로딩 중 오류 발생: {e}")
    return questions

def print_summary(results):
    """테스트 결과 요약을 출력합니다."""
    print("\n\n===== 최종 테스트 결과 요약 =====")
    
    # 모델별, 지표별 점수 집계
    summary = {}
    for result in results:
        model = result['model']
        if model not in summary:
            summary[model] = {
                'Faithfulness': [],
                'Accuracy': [],
                'Relevance': [],
                'Completeness': [],
                'count': 0
            }
        
        summary[model]['count'] += 1
        # 숫자로 변환 가능한 점수만 추가
        try:
            summary[model]['Faithfulness'].append(float(result['scores'].get("충실도", {}).get("score", 0)))
        except (ValueError, TypeError): pass
        try:
            summary[model]['Accuracy'].append(float(result['scores'].get("정확성", {}).get("score", 0)))
        except (ValueError, TypeError): pass
        try:
            summary[model]['Relevance'].append(float(result['scores'].get("관련성", {}).get("score", 0)))
        except (ValueError, TypeError): pass
        try:
            summary[model]['Completeness'].append(float(result['scores'].get("완결성", {}).get("score", 0)))
        except (ValueError, TypeError): pass

    # 평균 계산 및 출력
    for model, data in summary.items():
        count = data['count']
        if count == 0: continue
        
        avg_faith = sum(data['Faithfulness']) / len(data['Faithfulness']) if data['Faithfulness'] else 0
        avg_acc = sum(data['Accuracy']) / len(data['Accuracy']) if data['Accuracy'] else 0
        avg_rel = sum(data['Relevance']) / len(data['Relevance']) if data['Relevance'] else 0
        avg_comp = sum(data['Completeness']) / len(data['Completeness']) if data['Completeness'] else 0
        overall_avg = (avg_faith + avg_acc + avg_rel + avg_comp) / 4

        print(f"\n--- 모델: {model} (총 {count}개 질문) ---")
        print(f"  - 평균 충실도 (Faithfulness): {avg_faith:.2f} / 5")
        print(f"  - 평균 정확성 (Accuracy)    : {avg_acc:.2f} / 5")
        print(f"  - 평균 관련성 (Relevance)    : {avg_rel:.2f} / 5")
        print(f"  - 평균 완결성 (Completeness): {avg_comp:.2f} / 5")
        print(f"  ---------------------------------")
        print(f"  - 전체 평균 점수            : {overall_avg:.2f} / 5")

    print("\n===================================")
    print(f"상세 결과는 '{RESULTS_CSV_PATH}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    if initialize_vertexai() and load_vector_db():
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        questions_file = os.path.join(project_root, "evaluate", "rag_test", "question", "eval_questions_hardware_qa_30.jsonl")
        questions_to_test = load_questions_from_jsonl(questions_file, num_questions=10) # 10개의 질문으로 테스트

        if not questions_to_test:
            print("테스트할 질문이 없어 프로그램을 종료합니다.")
            sys.exit()
        
        test_run_id = f"Test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"\n===== 테스트 시작 (ID: {test_run_id}) =====")
        
        all_results = []
        for i, question in enumerate(questions_to_test):
            print(f"\n--- 질문 {i+1}/{len(questions_to_test)}: \"{question}\" ---")
            for model_id in MODEL_LIST:
                print(f"  -> 모델: {model_id} 처리 중...")
                prompt_type, context, answer = generate_answer_for_model(model_id, question)
                
                print(f"     - 답변 생성 완료. 평가를 시작합니다... (평가자: {EVALUATOR_MODEL})")
                scores = evaluate_response(question, answer, context, prompt_type)
                
                # 결과 저장을 위해 리스트에 추가
                all_results.append({
                    "model": model_id,
                    "scores": scores
                })

                # 상세 결과는 계속 CSV에 저장
                save_results_to_csv(test_run_id, question, model_id, prompt_type, context, answer, scores)
                print(f"     - 개별 결과 저장 완료.")
                
        print_summary(all_results)
        
    else:
        print("초기화 실패로 프로그램을 종료합니다.") 