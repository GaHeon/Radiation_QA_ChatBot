import concurrent.futures
import csv
from datetime import datetime
import json
import os
import re
import sys
import threading
import time
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
# 평가할 모델 목록
MODEL_LIST = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "claude-3-5-sonnet-v2"
]
# 평가자 모델
EVALUATOR_MODEL = "gemini-2.5-pro"
# 질문 요약용 경량 모델
HELPER_MODEL = "gemini-2.0-flash"

# 평가 질문 및 데이터 파일 경로
QUESTIONS_FILE = "eval_question_30.jsonl"
# 결과 CSV 파일이 저장될 디렉토리 (unified_test 안의 result 폴더)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# Vertex AI 및 DB 경로 설정
PROJECT_ID = "turing-berm-q3bf2"
LOCATION = "us-east5" # Claude 3.5 및 Gemini 2.5 Pro 사용 가능 리전

# --- 초기화 ---    
db = None
clients = {}

# .env 파일 로드를 위해 프로젝트 루트 경로 설정 및 로드
# 이 스크립트(run_evaluation.py)가 evaluate/unified_test/ 안에 있으므로,
# 프로젝트 루트(chatbot/)로 가려면 세 번 상위로 이동해야 합니다.
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

def generate_with_retry(client, model_name: str, prompt: str, max_retries: int = 3):
    """네트워크 오류나 빈 응답 발생 시 재시도 로직을 포함하여 콘텐츠를 생성합니다."""
    for attempt in range(max_retries):
        try:
            response_text = ""
            if model_name.startswith("claude"):
                response = client.messages.create(model=model_name, max_tokens=2048, messages=[{"role": "user", "content": prompt}])
                response_text = response.content[0].text
            else: # Gemini
                response = client.generate_content(prompt)
                # .text 접근 시 안전 필터링 등으로 비어있을 경우를 대비
                if response.candidates and response.candidates[0].content.parts:
                    response_text = response.candidates[0].content.parts[0].text
            
            if response_text.strip():
                return response_text.strip()
            else:
                print(f"  - 경고: 모델이 빈 응답을 반환했습니다. (시도 {attempt + 1}/{max_retries})")

        except exceptions.ServiceUnavailable as e:
            wait_time = 2 ** (attempt + 1)
            print(f"  - 경고: 서비스 일시 중단(503). {wait_time}초 후 재시도합니다... (시도 {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"  - 오류: 모델 응답 생성 중 예외 발생: {e}")
            # 다른 종류의 오류는 즉시 중단
            break
    
    print(f"  - 오류: 최대 재시도 횟수({max_retries}) 후에도 응답 생성 실패.")
    return f"모델 응답 생성 실패 (모델: {model_name})"

def condense_question(question: str):
    """LLM을 사용하여 질문을 간결한 검색어로 요약합니다."""
    client = get_client(HELPER_MODEL)
    prompt = f"""사용자의 질문을 벡터 검색에 가장 적합한 간결한 핵심 키워드 형태로 요약해주세요. 질문의 핵심 의도를 보존해야 합니다. 
문장이 아닌 키워드 구문(phrase)으로 만들어주세요. 원본 질문의 언어로 답변해주세요.
중요한 전문 용어나 제품명, 기술명은 반드시 포함해주세요.

---
Please summarize the following user's question into a concise keyword form suitable for vector search. The core intent of the question must be preserved. 
Create a keyword phrase, not a sentence. Respond in the original language of the question.
Important technical terms, product names, or technology names must be included.

Question / 질문: {question}

Summarized query / 요약된 검색어:"""
    condensed = generate_with_retry(client, HELPER_MODEL, prompt)
    if "모델 응답 생성 실패" in condensed or not condensed:
        print(f"  - 경고: 질문 요약 실패. 원본 질문을 검색에 사용합니다.")
        return question
    print(f"     - (Condense) 원본: '{question[:30]}...' -> 요약: '{condensed[:30]}...'")
    return condensed

def evaluate_multiple_responses(question, context, answers_with_models: list):
    """LLM을 사용하여 여러 모델의 답변을 한 번에 평가하고, 모델별 점수 딕셔너리를 반환합니다."""
    evaluator_client = get_client(EVALUATOR_MODEL)
    
    # 평가할 답변 목록을 프롬프트 형식에 맞게 구성
    answers_str = ""
    for i, (model_name, answer) in enumerate(answers_with_models):
        answers_str += f'---\n[Answer to Evaluate #{i+1} / 평가할 답변 #{i+1}]\n- Model / 모델명: "{model_name}"\n- Answer / 답변 내용: "{answer}"\n'

    prompt = f"""You are an evaluator who strictly assesses the quality of answers from multiple AI models at once.
The rationale for your evaluation should be in the same language as the user's question. You MUST respond ONLY in the JSON array format provided below.
(Korean) 당신은 여러 AI 모델의 답변 품질을 한 번에 엄격하게 평가하는 평가자입니다. 평가 근거는 사용자 질문과 같은 언어로 작성해야 합니다. 반드시 전체 결과를 JSON 배열 형식으로만 응답해야 합니다.

[User Question / 사용자 질문]:
{question}

[Reference Document / 참고 문서]:
{context}

[List of Answers to Evaluate / 평가할 답변 목록]:
{answers_str}---

[Evaluation Criteria / 평가 기준]
1. Faithfulness (충실도): Is the answer generated based solely on the 'Reference Document'?
2. Accuracy (정확성): Is the answer factually accurate?
3. Relevance (관련성): Is the answer relevant to the user's question?
4. Completeness (완결성): Does the answer include all key information from the 'Reference Document'?
5. Domain Appropriateness (전문성): Does the answer use terminology and knowledge appropriate for an expert in the field?

[Output Format / 출력 형식]
```json
[
  {{
    "model_name": "evaluated_model_name_1",
    "scores": {{
      "Faithfulness": {{"score": [0-5], "reason": "Rationale for evaluation"}},
      "Accuracy": {{"score": [0-5], "reason": "Rationale for evaluation"}},
      "Relevance": {{"score": [0-5], "reason": "Rationale for evaluation"}},
      "Completeness": {{"score": [0-5], "reason": "Rationale for evaluation"}},
      "Domain Appropriateness": {{"score": [0-5], "reason": "Rationale for evaluation"}}
    }}
  }},
  {{
    "model_name": "evaluated_model_name_2",
    "scores": {{ "Faithfulness": {{...}}, "Accuracy": {{...}}, ... }}
  }}
]
```"""

    evaluation_text = generate_with_retry(evaluator_client, EVALUATOR_MODEL, prompt)
    match = re.search(r"```json\s*(\[.*?\])\s*```", evaluation_text, re.DOTALL)
    if match:
        try:
            evaluations = json.loads(match.group(1))
            return {item['model_name']: item['scores'] for item in evaluations if 'model_name' in item}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  - 경고: 평가자 모델의 JSON 배열 응답 파싱 실패: {e}")
    return {}

def extract_json_from_string(s: str) -> dict:
    """문자열에서 JSON 블록을 추출하여 파싱합니다."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            print("  - 경고: 평가자 모델의 JSON 응답 파싱 실패")
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
        5. 전문성 (Domain Appropriateness): 답변이 해당 분야(예: 의료기기)의 전문가 수준에 맞는 용어와 지식을 사용했는가?
        [평가할 데이터]
        - 사용자 질문: "{question}"
        - 참고 문서: "{context}"
        - 평가할 답변: "{answer}"
        [출력 형식]
        ```json
        {{"충실도": {{"score": [0-5], "reason": "평가 근거"}}, "정확성": {{"score": [0-5], "reason": "평가 근거"}}, "관련성": {{"score": [0-5], "reason": "평가 근거"}}, "완결성": {{"score": [0-5], "reason": "평가 근거"}}, "전문성": {{"score": [0-5], "reason": "평가 근거"}}}}
        ```"""
    else:  # Fallback
        prompt = f"""당신은 답변의 품질을 엄격하게 평가하는 평가자입니다. 다음 기준에 따라 0점에서 5점 사이로 채점하고, 평가 근거를 한국어로 간략하게 작성해주세요. 반드시 JSON 형식으로만 응답해야 합니다.
        [평가 기준]
        1. 충실도 (Faithfulness): 일반 지식 기반 답변이므로 이 항목은 5점으로 고정합니다.
        2. 정확성 (Accuracy): 답변이 일반적으로 알려진 사실에 비추어 정확한가?
        3. 관련성 (Relevance): 답변이 사용자의 질문 의도와 얼마나 관련이 있는가?
        4. 완결성 (Completeness): 답변이 질문의 모든 측면을 충분히 다루고 있는가?
        5. 전문성 (Professionalism): 답변이 일반적인 주제에 대해서도 깊이 있고 전문적인 어조로 설명하는가?
        [평가할 데이터]
        - 사용자 질문: "{question}"
        - 평가할 답변: "{answer}"
        [출력 형식]
        ```json
        {{"충실도": {{"score": 5, "reason": "일반 지식 답변"}}, "정확성": {{"score": [0-5], "reason": "평가 근거"}}, "관련성": {{"score": [0-5], "reason": "평가 근거"}}, "완결성": {{"score": [0-5], "reason": "평가 근거"}}, "전문성": {{"score": [0-5], "reason": "평가 근거"}}}}
        ```"""
    
    evaluation_text = generate_with_retry(evaluator_client, EVALUATOR_MODEL, prompt)
    return extract_json_from_string(evaluation_text)

def save_results_to_csv(test_run_id, question, model_name, prompt_type, answer, scores):
    """결과를 CSV 파일에 스레드 안전하게 저장합니다."""
    # 결과를 저장할 디렉토리 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RESULTS_DIR, f'{test_run_id}.csv')
    
    # 파일이 없으면 헤더를 추가
    write_header = not os.path.exists(file_path)
    
    # 헤더 정의 (Context 제거)
    headers = [
        "Test_ID", "Question", "Model", "Prompt_Type", "Answer",
        "Faithfulness_Score", "Faithfulness_Reason",
        "Accuracy_Score", "Accuracy_Reason",
        "Relevance_Score", "Relevance_Reason",
        "Completeness_Score", "Completeness_Reason",
        "Domain_Appropriateness_Score", "Domain_Appropriateness_Reason"
    ]
    
    # 데이터 행 구성 (Context 제거)
    row = {
        "Test_ID": test_run_id,
        "Question": question,
        "Model": model_name,
        "Prompt_Type": prompt_type,
        "Answer": answer,
        "Faithfulness_Score": scores.get("Faithfulness", {}).get("score"),
        "Faithfulness_Reason": scores.get("Faithfulness", {}).get("reason"),
        "Accuracy_Score": scores.get("Accuracy", {}).get("score"),
        "Accuracy_Reason": scores.get("Accuracy", {}).get("reason"),
        "Relevance_Score": scores.get("Relevance", {}).get("score"),
        "Relevance_Reason": scores.get("Relevance", {}).get("reason"),
        "Completeness_Score": scores.get("Completeness", {}).get("score"),
        "Completeness_Reason": scores.get("Completeness", {}).get("reason"),
        "Domain_Appropriateness_Score": scores.get("Domain Appropriateness", {}).get("score"),
        "Domain_Appropriateness_Reason": scores.get("Domain Appropriateness", {}).get("reason")
    }
    
    try:
        with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerow([row[h] for h in headers])
    except Exception as e:
        print(f"  - 경고: CSV 파일 저장 중 오류 발생: {e}")

def load_questions_from_jsonl(file_path, num_questions=10):
    """JSONL 파일에서 질문을 로드합니다."""
    questions = []
    print(f"질문 파일 로딩 시작: {file_path}")
    if not os.path.exists(file_path):
        print(f"오류: 질문 파일을 찾을 수 없습니다: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_questions: break
                questions.append(json.loads(line)['question'])
        print(f"총 {len(questions)}개의 질문을 로드했습니다.")
    except Exception as e:
        print(f"질문 파일 로딩 중 오류 발생: {e}")
    return questions

def print_summary(results):
    """테스트 결과 요약을 출력합니다."""
    print("\n\n===== 최종 테스트 결과 요약 =====")
    summary = {}
    for result in results:
        model = result['model']
        if model not in summary:
            summary[model] = {'Faithfulness': [], 'Accuracy': [], 'Relevance': [], 'Completeness': [], 'Domain Appropriateness': [], 'count': 0}
        summary[model]['count'] += 1
        try: summary[model]['Faithfulness'].append(float(result['scores'].get("Faithfulness", {}).get("score", 0)))
        except (ValueError, TypeError): pass
        try: summary[model]['Accuracy'].append(float(result['scores'].get("Accuracy", {}).get("score", 0)))
        except (ValueError, TypeError): pass
        try: summary[model]['Relevance'].append(float(result['scores'].get("Relevance", {}).get("score", 0)))
        except (ValueError, TypeError): pass
        try: summary[model]['Completeness'].append(float(result['scores'].get("Completeness", {}).get("score", 0)))
        except (ValueError, TypeError): pass
        try: summary[model]['Domain Appropriateness'].append(float(result['scores'].get("Domain Appropriateness", {}).get("score", 0)))
        except (ValueError, TypeError): pass

    for model, data in summary.items():
        count = data['count']
        if count == 0: continue
        avg_faith = sum(data['Faithfulness']) / len(data['Faithfulness']) if data['Faithfulness'] else 0
        avg_acc = sum(data['Accuracy']) / len(data['Accuracy']) if data['Accuracy'] else 0
        avg_rel = sum(data['Relevance']) / len(data['Relevance']) if data['Relevance'] else 0
        avg_comp = sum(data['Completeness']) / len(data['Completeness']) if data['Completeness'] else 0
        avg_prof = sum(data['Domain Appropriateness']) / len(data['Domain Appropriateness']) if data['Domain Appropriateness'] else 0
        overall_avg = (avg_faith + avg_acc + avg_rel + avg_comp + avg_prof) / 5

        print(f"\n--- 모델: {model} (총 {count}개 질문) ---")
        print(f"  - 평균 충실도 (Faithfulness): {avg_faith:.2f} / 5")
        print(f"  - 평균 정확성 (Accuracy)    : {avg_acc:.2f} / 5")
        print(f"  - 평균 관련성 (Relevance)    : {avg_rel:.2f} / 5")
        print(f"  - 평균 완결성 (Completeness): {avg_comp:.2f} / 5")
        print(f"  - 평균 전문성 (Domain Approp.): {avg_prof:.2f} / 5")
        print(f"  ---------------------------------")
        print(f"  - 전체 평균 점수            : {overall_avg:.2f} / 5")

    print("\n===================================")
    # 상세 결과 파일 경로 안내를 test_run_id에 맞춰 동적으로 변경
    first_result_path = os.path.join(RESULTS_DIR, f"Test-*.csv")
    print(f"상세 결과는 '{RESULTS_DIR}' 폴더에 테스트 ID별 CSV 파일로 저장되었습니다. (예: {first_result_path})")

def prepare_data_for_evaluation(question: str):
    """(1단계) 질문 요약, 문서 검색, 프롬프트 타입 결정을 처리합니다."""
    print(f"  - 데이터 준비: '{question[:30]}...'")

    # 1. 질문 요약
    condensed_query = condense_question(question)

    # 2. 하이브리드 검색: 요약된 질문과 원본 질문 모두로 검색
    docs_with_scores_condensed = db.similarity_search_with_score(condensed_query, k=8)
    docs_with_scores_original = db.similarity_search_with_score(question, k=8)
    
    # 3. 두 검색 결과를 합치고 중복 제거 (더 나은 결과 선택)
    all_docs = []
    seen_contents = set()
    
    # 요약된 질문 결과 먼저 추가
    for doc, score in docs_with_scores_condensed:
        if doc.page_content[:100] not in seen_contents:
            all_docs.append((doc, score, "condensed"))
            seen_contents.add(doc.page_content[:100])
    
    # 원본 질문 결과 추가 (더 나은 점수면 교체)
    for doc, score in docs_with_scores_original:
        content_key = doc.page_content[:100]
        if content_key not in seen_contents:
            all_docs.append((doc, score, "original"))
            seen_contents.add(content_key)
        else:
            # 이미 있는 경우 더 나은 점수로 교체
            for i, (existing_doc, existing_score, _) in enumerate(all_docs):
                if existing_doc.page_content[:100] == content_key and score < existing_score:
                    all_docs[i] = (doc, score, "original")
                    break
    
    # 점수로 정렬
    all_docs.sort(key=lambda x: x[1])
    
    # 4. RAG 또는 Fallback 프롬프트 결정 (임계값을 0.9로 더 완화)
    if all_docs and all_docs[0][1] <= 0.9: # 임계값 0.8 -> 0.9로 더 완화
        prompt_type = "RAG (Document-based)"
        # 더 많은 문서를 포함하고 유사도가 높은 문서들만 필터링
        relevant_docs = [(doc, score) for doc, score, _ in all_docs if score <= 0.95]
        if relevant_docs:
            context = "\n\n".join(f"문서 {i+1} (유사도: {score:.4f}):\n{doc.page_content}" 
                                  for i, (doc, score) in enumerate(relevant_docs))
        else:
            # 유사도가 높아도 최상위 문서는 포함
            context = "\n\n".join(f"문서 {i+1} (유사도: {score:.4f}):\n{doc.page_content}" 
                                  for i, (doc, score, _) in enumerate(all_docs[:5]))
    else:
        prompt_type = "Fallback (General Knowledge)"
        context = "N/A"
    
    return {"question": question, "context": context, "prompt_type": prompt_type}

def generate_single_answer_task(args):
    """단일 모델에 대한 답변 생성을 위한 작은 작업 함수"""
    model_id, prompt = args
    client = get_client(model_id)
    answer = generate_with_retry(client, model_id, prompt)
    return model_id, answer

def evaluate_question_across_models(args):
    """하나의 질문에 대해 모든 모델의 답변 생성과 '일괄 평가'를 처리합니다."""
    prepared_data, test_run_id, csv_lock = args
    question = prepared_data['question']
    context = prepared_data['context']
    prompt_type = prepared_data['prompt_type']
    
    print(f"\n-> 질문 처리 시작: '{question[:30]}...'")

    # 1. 답변 생성 프롬프트 구성
    if prompt_type == "RAG (Document-based)":
        prompt = f"""You are an expert AI assistant that answers based on the provided documents. Respond in the same language as the User Question.
(Korean) 당신은 제공된 문서를 기반으로 답변하는 전문 AI 어시스턴트입니다. 사용자 질문과 같은 언어로 답변해주세요.

[Reference Documents / 참고 문서]:
---
{context}
---
[User Question / 사용자 질문]: {question}

[Response Guidelines / 응답 지침]:
1. You MUST answer using ONLY the 'Reference Documents' provided. Do not use prior knowledge.
(Korean) 반드시 위에 제공된 '참고 문서'만을 사용하여 답변해야 합니다. 사전 지식을 사용해서는 안 됩니다.
2. If the documents contain ANY relevant information (even partial), use it to provide the best possible answer.
(Korean) 만약 문서에 관련 정보가 부분적으로라도 있다면, 그것을 사용하여 최선의 답변을 제공하세요.
3. If the documents do not contain any relevant information at all, answer ONLY with "I could not find information about the question in the provided documents."
(Korean) 만약 문서에 질문과 관련된 내용이 전혀 없다면, "제공된 문서에서는 해당 질문에 대한 정보를 찾을 수 없습니다." 라고만 답변하세요.
4. Be creative in connecting related information from the documents to answer the question.
(Korean) 문서의 관련 정보를 창의적으로 연결하여 질문에 답변하세요.
5. Quote or reference specific parts of the documents when possible to support your answer.
(Korean) 가능한 경우 문서의 특정 부분을 인용하거나 참조하여 답변을 뒷받침하세요.
6. Even if the documents only contain tangential information, try to use it to provide a relevant answer.
(Korean) 문서에 간접적인 정보만 있더라도, 그것을 사용하여 관련성 있는 답변을 제공하세요.

[Answer / 답변]:"""
    else: # Fallback
        prompt = f"""You are a helpful AI assistant. Answer based on your general knowledge. Respond in the same language as the User Question.
(Korean) 당신은 유능한 AI 어시스턴트입니다. 일반 지식을 바탕으로 사용자 질문과 같은 언어로 답변해주세요.

[User Question / 사용자 질문]: {question}

[Response Guidelines / 응답 지침]:
1. Begin your answer by stating, "I couldn't find relevant information in my documents, but according to general knowledge,"
(Korean) 답변 시작 시, "제가 가진 문서에서는 관련 정보를 찾지 못했지만, 일반적인 지식에 따르면" 이라고 명시하세요.

[Answer / 답변]:"""

    # 2. 모든 모델에 대한 답변 동시 생성
    answers_with_models = []
    generation_tasks = [(model_id, prompt) for model_id in MODEL_LIST]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_LIST)) as executor:
        for model_id, answer in executor.map(generate_single_answer_task, generation_tasks):
            if "모델 응답 생성 실패" not in answer:
                answers_with_models.append((model_id, answer))

    if not answers_with_models:
        print(f"  - 모든 모델이 답변 생성에 실패하여 질문을 건너뜁니다: '{question[:30]}...'")
        return []

    # 3. 생성된 모든 답변을 '일괄 평가' (API 호출 1번)
    print(f"  - 일괄 평가 시작: {len(answers_with_models)}개 답변에 대해...")
    batch_eval_results = evaluate_multiple_responses(question, context, answers_with_models)

    # 4. 개별 결과 저장 및 반환
    results_for_this_question = []
    for model_id, answer in answers_with_models:
        scores = batch_eval_results.get(model_id, {})
        with csv_lock:
            save_results_to_csv(test_run_id, question, model_id, prompt_type, answer, scores)
        results_for_this_question.append({"model": model_id, "scores": scores})
    
    print(f"<- 질문 처리 완료: '{question[:30]}...'")
    return results_for_this_question

if __name__ == "__main__":
    if initialize_vertexai() and load_vector_db():
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        questions_file_path = os.path.join(script_dir, QUESTIONS_FILE)
        questions = load_questions_from_jsonl(questions_file_path)

        if not questions:
            print("테스트할 질문이 없어 프로그램을 종료합니다.")
            sys.exit()

        # --- 1단계: 질문 요약 및 문서 검색 병렬 처리 ---
        print("\n===== 1단계: 질문 요약 및 문서 검색 병렬 처리 시작... =====")
        prepared_data_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            data_iterator = tqdm(executor.map(prepare_data_for_evaluation, questions), total=len(questions), desc="데이터 준비 진행률")
            prepared_data_list = list(data_iterator)
        print("===== 1단계: 질문 요약 및 문서 검색 병렬 처리 완료 =====")

        # --- 2단계: 질문별 병렬 평가 ---
        test_run_id = f"Test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"\n===== 2단계: 모델 평가 병렬 실행 시작 (ID: {test_run_id}) =====")
        print(f"총 {len(prepared_data_list)}개 질문에 대해 일괄 평가를 병렬로 처리합니다.")
        
        all_results = []
        csv_lock = threading.Lock()
        
        tasks = [(data, test_run_id, csv_lock) for data in prepared_data_list]

        # 동시 작업자 수를 10 -> 3으로 줄여 API 과부하 방지
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results_iterator = tqdm(executor.map(evaluate_question_across_models, tasks), total=len(tasks), desc="전체 질문 평가 진행률")
            for res_list in results_iterator:
                all_results.extend(res_list)
                
        print_summary(all_results)
    else:
        print("초기화 실패로 프로그램을 종료합니다.") 