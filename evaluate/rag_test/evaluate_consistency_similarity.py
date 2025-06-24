
import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.generative_models import GenerativeModel

QUESTIONS = [
    "Focal spot 검사 기준은 무엇인가요?",
    "CT의 slice thickness 기준 오차 범위는 얼마인가요?",
    "DR 시스템에서 고정 SID 검사 방법은?",
    "X선 발생장치에서 과열 방지를 위한 조치는?",
    "CR과 DR의 QA 항목 차이는 무엇인가요?"
]

def get_response(prompt: str):
    model = GenerativeModel("gemini-2.0-flash")
    responses = model.generate_content(
        contents=[prompt],
        stream=False
    )
    return responses.text.strip()

def get_avg_similarity(responses):
    if len(responses) < 2:
        return 1.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(responses)
    sim_matrix = cosine_similarity(tfidf_matrix)
    # 상삼각행렬 평균 계산 (자기 자신 제외)
    n = len(responses)
    sims = [sim_matrix[i][j] for i in range(n) for j in range(i+1, n)]
    return float(np.mean(sims))

results = []

for question in QUESTIONS:
    print(f"질문: {question}")
    prompt = f"""
당신은 방사선 QA 분야의 전문가 챗봇입니다.
다음 질문에 대해 정확하고 실무적인 설명을 해주세요.

질문: {question}
답변:
"""

    responses = []
    for _ in range(5):
        resp = get_response(prompt)
        responses.append(resp)
        time.sleep(1)

    avg_sim = get_avg_similarity(responses)
    is_consistent = avg_sim >= 0.85

    results.append({
        "question": question,
        "responses": responses,
        "average_similarity": round(avg_sim, 4),
        "is_consistent": is_consistent
    })

    print(f"유사도 평균: {avg_sim:.4f} → {'✅ 일관성 있음' if is_consistent else '❌ 일관성 낮음'}")
    print("=" * 60)

# 결과 저장
output_path = "/mnt/data/evaluate_consistency_similarity.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {output_path}")
