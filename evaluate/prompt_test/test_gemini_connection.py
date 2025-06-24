import os
import sys
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel

# --- 초기화 ---
# 상위 디렉토리의 .env 파일을 찾기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

def test_gemini_1_5_pro_connection():
    """
    vertexai SDK를 사용하여 Gemini 1.5 Pro 모델 연결을 테스트합니다.
    (메인 평가 스크립트와 동일한 방식)
    """
    # --- 설정 ---
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = "us-east5"  # 메인 스크립트와 동일한 리전 사용
    MODEL_NAME = "gemini-2.5-pro" # 최신 모델인 Gemini 2.5 Pro로 변경

    if not PROJECT_ID:
        print("오류: .env 파일에 PROJECT_ID를 설정해주세요.")
        return

    print("--- Gemini 2.5 Pro 연결 테스트 시작 (vertexai SDK) ---")
    print(f"  - Project ID: {PROJECT_ID}")
    print(f"  - Location: {LOCATION}")
    print(f"  - Model: {MODEL_NAME}")
    print("----------------------------------------------------")

    try:
        # Vertex AI 초기화
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # 모델 초기화
        model = GenerativeModel(MODEL_NAME)

        # 간단한 프롬프트로 응답 생성 테스트
        prompt = "대한민국의 수도는 어디인가요?"
        print(f"\n> 테스트 프롬프트: \"{prompt}\"")
        
        # 응답 생성
        response = model.generate_content(prompt)
        
        print("\n> 모델 응답:")
        print("------------------")
        print(response.text.strip())
        print("------------------")
        print("\n[성공] Gemini 2.5 Pro 모델과 성공적으로 통신했습니다.")

    except Exception as e:
        print(f"\n[실패] 모델 연결 또는 응답 생성 중 오류가 발생했습니다.")
        print(f"  - 오류 상세 정보: {e}")
        print("\n  - 확인 사항:")
        print("    1. gcloud CLI에 올바르게 인증되었는지 확인하세요. (gcloud auth application-default login)")
        print(f"    2. Vertex AI API가 '{PROJECT_ID}' 프로젝트에서 활성화되었는지 확인하세요.")
        print(f"    3. '{LOCATION}' 리전에서 '{MODEL_NAME}' 모델을 사용할 권한이 있는지 확인하세요.")


if __name__ == "__main__":
    test_gemini_1_5_pro_connection() 