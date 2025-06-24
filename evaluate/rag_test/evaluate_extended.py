import os
import json
import time
import numpy as np
from vertexai.generative_models import GenerativeModel
from google.api_core import exceptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import concurrent.futures

# Configuration
# 평가 질문 파일 경로
EVAL_QUESTIONS_FILE = 'eval_questions_hardware_qa_30.jsonl'
# 평가 결과 저장 경로
EVAL_RESULTS_FILE = 'evaluate_results_hardware_qa.json'

# --- 1. 모델 및 DB 로드 ---
print("임베딩 모델 및 벡터 DB 로딩 시작...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
)
# 스크립트의 위치를 기준으로 프로젝트 루트를 찾아 embed_faiss 폴더 경로를 설정합니다.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
local_path = os.path.join(project_root, "embed_faiss")

if not os.path.exists(local_path):
    raise FileNotFoundError(f"Embeddings directory not found at: {local_path}. Please ensure 'embed_faiss' directory is in the project root.")

db = FAISS.load_local(local_path, embeddings, allow_dangerous_deserialization=True)
print("임베딩 모델 및 벡터 DB 로딩 완료.")

def generate_with_retry(model, prompt: str, max_retries: int = 3):
    """
    네트워크 오류나 빈 응답 발생 시 재시도 로직을 포함하여 콘텐츠를 생성합니다.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = model.generate_content(prompt, stream=False)
            # .text에 접근 시 ValueError가 발생하면 응답이 비었거나 차단된 것입니다.
            # 이 접근 자체가 유효성 검사 역할을 합니다.
            return response.text.strip()
        except exceptions.ServiceUnavailable as e:
            retries += 1
            wait_time = 2 ** retries  # 2, 4, 8초 후 재시도
            print(f"\nWarning: Service Unavailable. {wait_time}초 후 재시도합니다... (시도 {retries}/{max_retries})")
            time.sleep(wait_time)
        except ValueError:
            # 콘텐츠가 안전 설정 등에 의해 필터링되어 .text 접근 시 오류가 발생하는 경우
            print(f"\nWarning: 모델이 빈 응답을 반환했습니다 (Safety/Filtering). 해당 항목을 건너뜁니다.")
            return ""
        except Exception as e:
            # 다른 종류의 예기치 않은 오류 (e.g. AttributeError)
            print(f"\n콘텐츠 생성 중 예기치 않은 오류 발생: {e}")
            return ""
    print(f"\nError: 최대 재시도 횟수({max_retries})에 도달했습니다. 생성을 건너뜁니다.")
    return ""

def load_eval_questions():
    """질문 파일을 로드합니다."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 설정에 지정된 질문 파일을 사용하도록 경로 수정
        file_path = os.path.join(script_dir, EVAL_QUESTIONS_FILE)
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: 질문 파일을 찾을 수 없습니다. 경로: {EVAL_QUESTIONS_FILE}")
        return []

def get_similarity_score(answer, reference):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([answer, reference])
    sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(sim_matrix[0][0])

def get_judge_evaluation(question, answer, context):
    model = GenerativeModel("gemini-2.0-flash")
    prompt = f"""
다음은 사용자의 질문, 챗봇의 응답, 그리고 응답 생성의 근거가 된 참고 문서입니다.

[질문]:
{question}

[참고 문서]:
{context}

[챗봇 응답]:
{answer}

다음 기준에 따라 0~5점으로 엄격하게 평가하세요:

1. 정확성 (Accuracy): [챗봇 응답]이 사실과 얼마나 일치하는지.
2. 충실도 (Faithfulness): [챗봇 응답]이 **오직 [참고 문서]에 명시된 정보만을 사용**하여 생성되었는지. **문서에 없는 내용, 외부 지식, 또는 과장된 해석이 포함되었다면 0점을 부여하세요.**
3. 관련성 (Relevance): [챗봇 응답]이 [질문]의 의도와 얼마나 관련 있는지.
4. 전문성 (Domain Appropriateness): 방사선 QA 도메인에 적절한 표현과 지식을 사용했는지.
5. 표현력 (Fluency): 문장이 자연스럽고 명확하며 매끄럽게 전달되는지.

다음과 같은 JSON 형식으로 응답하세요:
{{
  "정확성": 0~5,
  "충실도": 0~5,
  "관련성": 0~5,
  "전문성": 0~5,
  "표현력": 0~5
}}
"""
    judge_text = generate_with_retry(model, prompt)
    return judge_text

def translate_query_to_english(model, query: str):
    """LLM을 사용하여 한글 쿼리를 영어로 번역합니다."""
    prompt = f"Translate the following Korean text to English. Do not add any extra explanation. Just provide the translated text.\n\nKorean: {query}\n\nEnglish:"
    
    translated_text = generate_with_retry(model, prompt)
    
    # 번역 실패 시 원본 쿼리 사용
    if not translated_text:
        print(f"\nWarning: Query translation failed for '{query}'. Using original query.")
        return query
    
    return translated_text

def generate_hypothetical_answer(model, question: str):
    """
    HyDE (Hypothetical Document Embeddings)를 위해 가상의 답변을 생성합니다.
    """
    prompt = f"""다음 질문에 대해 이상적인 답변을 생성해주세요. 이 답변은 사실이 아니어도 괜찮습니다. 
오직 벡터 검색의 품질을 높이기 위한 목적으로만 사용됩니다. 답변은 상세하고 명확하게 작성해주세요.

질문: {question}

답변:
"""
    hypothetical_answer = generate_with_retry(model, prompt)
    if not hypothetical_answer:
        print(f"\nWarning: 가상 답변 생성 실패: '{question}'. 원본 질문을 대신 사용합니다.")
        return question
    return hypothetical_answer

def extract_scores(text):
    text = re.sub(r'```json\s*|\s*```', '', text)
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        print(f"Error: 응답에서 JSON 객체를 찾을 수 없습니다.\n응답 내용: {text}")
        return {"정확성": 0, "충실도": 0, "관련성": 0, "전문성": 0, "표현력": 0}

    json_str = json_match.group(0)
    try:
        result = json.loads(json_str)
        return {
            "정확성": result.get("정확성", 0),
            "충실도": result.get("충실도", 0),
            "관련성": result.get("관련성", 0),
            "전문성": result.get("전문성", 0),
            "표현력": result.get("표현력", 0)
        }
    except json.JSONDecodeError as e:
        print(f"Error: JSON 파싱에 실패했습니다.\n파싱 대상: {json_str}\n에러: {e}")
        return {"정확성": 0, "충실도": 0, "관련성": 0, "전문성": 0, "표현력": 0}

def process_question(item):
    """
    하나의 질문 항목에 대한 전체 RAG 평가 파이프라인을 처리합니다.
    (문서 검색 -> 답변 생성 -> 답변 평가)
    """
    # 각 스레드에서 모델 인스턴스를 안전하게 생성합니다.
    model = GenerativeModel("gemini-2.0-flash")
    
    question = item["question"]
    reference = item.get("answer", "")

    # 1. HyDE: 검색 품질 향상을 위해 가상의 답변 생성
    hypothetical_answer = generate_hypothetical_answer(model, question)
    if hypothetical_answer != question:
        tqdm.write(f"[HyDE] 가상 답변 생성 완료 for '{question}'")

    # 2. 가상 답변을 사용하여 유사도 검색 수행 (k=10으로 늘려서 더 많은 context 확보)
    docs = db.similarity_search(hypothetical_answer, k=10)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. 답변 생성 시에는 원본 한글 질문 사용
    user_prompt = f"""당신은 주어진 [참고 문서]를 바탕으로 답변을 생성하는 방사선 장비 품질관리(QA) 전문가 AI입니다.

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

[답변]:
"""
    response_text = generate_with_retry(model, user_prompt)
    if not response_text:
        print(f"\n질문({question})에 대한 답변 생성 실패. 해당 항목을 건너뜁니다.")
        return None # 실패 시 None 반환

    judge_result = get_judge_evaluation(question, response_text, context)
    if not judge_result:
        print(f"\n질문({question})에 대한 평가 실패. 해당 항목을 건너뜁니다.")
        score_dict = {"정확성": 0, "충실도": 0, "관련성": 0, "전문성": 0, "표현력": 0}
    else:
        score_dict = extract_scores(judge_result)
    
    sim_score = get_similarity_score(response_text, reference) if reference else None

    return {
        "question": question,
        "response": response_text,
        "scores": score_dict,
        "similarity": round(sim_score, 4) if sim_score else "N/A"
    }

def evaluate():
    questions = load_eval_questions()
    if not questions:
        print("평가할 질문이 없습니다. 스크립트를 종료합니다.")
        return

    results = []
    
    # 모든 점수를 저장할 딕셔너리 초기화
    all_scores = {
        "정확성": [], "충실도": [], "관련성": [], "전문성": [], "표현력": []
    }
    all_similarities = []
    not_found_count = 0

    # ThreadPoolExecutor를 사용하여 병렬로 질문 처리 (최대 10개 동시 작업)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 작업을 제출하고 future 객체를 받음
        future_to_question = {executor.submit(process_question, item): item for item in questions}
        
        # tqdm을 사용하여 진행률 표시
        for future in tqdm(concurrent.futures.as_completed(future_to_question), total=len(questions), desc="Evaluating RAG pipeline"):
            result = future.result()
            if result:
                results.append(result)

    # 모든 결과가 수집된 후 점수 집계
    for res in results:
        score_dict = res['scores']
        for key in all_scores:
            all_scores[key].append(score_dict.get(key, 0))
        
        if res['similarity'] != "N/A":
            all_similarities.append(res['similarity'])
        
        # "정보를 찾을 수 없음" 응답 카운트
        if "정보를 찾을 수 없습니다" in res.get("response", ""):
            not_found_count += 1

    # 결과 파일을 evaluate 폴더 내에 저장하도록 경로 수정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "evaluate_results_answerable.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"📄 평가 완료: {output_path}")

    # --- 최종 평가 결과 (평균 점수) ---
    print("\n--- 최종 평가 결과 (평균 점수) ---")
    avg_scores = {}
    for key, scores in all_scores.items():
        avg_scores[key] = np.mean(scores) if scores else 0
        print(f"- {key}: {avg_scores[key]:.2f} / 5")

    if all_similarities:
        avg_similarity = np.mean(all_similarities)
        print(f"- 유사도 (Similarity): {avg_similarity:.4f}")

    if questions:
        not_found_rate = (not_found_count / len(questions)) * 100
        print(f"📊 검색 실패율: {not_found_rate:.2f}% ({not_found_count}/{len(questions)})")
    
    print("---------------------------------")

    # 평가 결과를 파일에 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 설정에 지정된 결과 파일을 사용하도록 경로 수정
    results_path = os.path.join(script_dir, EVAL_RESULTS_FILE)
    
    final_summary_with_results = {
        "summary": {
            "정확성": avg_scores["정확성"],
            "충실도": avg_scores["충실도"],
            "관련성": avg_scores["관련성"],
            "전문성": avg_scores["전문성"],
            "표현력": avg_scores["표현력"]
        },
        "individual_results": results
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_summary_with_results, f, ensure_ascii=False, indent=4)
    
    print(f"\n평가 결과가 {results_path} 에 저장되었습니다.")

if __name__ == "__main__":
    evaluate()
