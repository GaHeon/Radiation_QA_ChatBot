import os
import sys
import json
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel
import vertexai
import concurrent.futures
from tqdm import tqdm

# --- 설정 ---
# 가상 답변 생성에 사용할 모델
HELPER_MODEL = "gemini-2.5-pro" 
# 입력 질문 파일
INPUT_QUESTIONS_FILE = 'eval_question_30.jsonl'
# 가상 답변이 추가된 출력 파일
OUTPUT_FILE = 'questions_with_hyde.jsonl'
# 프로젝트 ID 및 리전
PROJECT_ID = None
LOCATION = "us-east5"

# --- 초기화 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()
clients = {} # 모델 클라이언트 캐싱

def initialize_vertexai():
    """Vertex AI를 초기화합니다."""
    global PROJECT_ID
    PROJECT_ID = os.getenv("PROJECT_ID")
    if not PROJECT_ID:
        print("오류: .env 파일에 PROJECT_ID를 설정해야 합니다.")
        sys.exit(1)
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print(f"Vertex AI 초기화 완료. (Project: {PROJECT_ID}, Location: {LOCATION})")
    return True

def get_client(model_name: str):
    """Vertex AI 모델 클라이언트를 로드하고 캐싱합니다."""
    if model_name not in clients:
        clients[model_name] = GenerativeModel(model_name)
    return clients[model_name]

def generate_hypothetical_answer(question: str):
    """HyDE를 위해 LLM을 사용하여 가상의 답변을 생성합니다."""
    client = get_client(HELPER_MODEL)
    prompt = f"""다음 질문에 대해 이상적인 답변을 생성해주세요. 이 답변은 사실이 아니어도 괜찮습니다. 
오직 벡터 검색의 품질을 높이기 위한 목적으로만 사용됩니다. 답변은 상세하고 명확하게 작성해주세요.

질문: {question}

이상적인 답변:"""
    try:
        response = client.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"오류: 가상 답변 생성 중 예외 발생 - {question[:30]}... ({e})")
    return question # 실패 시 원본 질문 반환

def process_question(question_text):
    """단일 질문에 대해 가상 답변을 생성하고 딕셔너리로 반환합니다."""
    hyde_answer = generate_hypothetical_answer(question_text)
    return {"question": question_text, "hyde_answer": hyde_answer}

def main():
    """메인 실행 함수"""
    if not initialize_vertexai():
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_QUESTIONS_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    # 1. 원본 질문 로드
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            questions = [json.loads(line)['question'] for line in f]
        print(f"총 {len(questions)}개의 질문을 로드했습니다: {input_path}")
    except FileNotFoundError:
        print(f"오류: 입력 질문 파일을 찾을 수 없습니다: {input_path}")
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"오류: 질문 파일 처리 중 오류 발생: {e}")
        return
        
    # 2. 가상 답변 병렬 생성
    print("\n===== 가상 답변(HyDE) 생성 시작... =====")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_question = {executor.submit(process_question, q): q for q in questions}
        for future in tqdm(concurrent.futures.as_completed(future_to_question), total=len(questions), desc="HyDE 답변 생성 중"):
            results.append(future.result())
            
    # 3. 결과 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\n===== 가상 답변 생성 완료 =====")
    print(f"결과가 다음 파일에 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main() 