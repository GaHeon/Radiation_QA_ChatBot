import os
from dotenv import load_dotenv
from anthropic import AnthropicVertex

def test_claude_connection():
    """
    Vertex AI를 통해 Claude 모델에 대한 연결을 테스트하는 최소 기능 스크립트.
    """
    print("--- Claude 연결 테스트 시작 ---")
    
    # 1. 환경 변수 로드
    load_dotenv()
    project_id = os.getenv("PROJECT_ID")
    # Vertex AI에서 Claude 3.5 Sonnet을 지원하는 리전
    location = "us-east5" 
    
    if not project_id:
        print("오류: .env 파일에서 PROJECT_ID를 찾을 수 없습니다.")
        return

    print(f"Project ID: {project_id}")
    print(f"Location: {location}")

    try:
        # 2. Anthropic Vertex 클라이언트 초기화
        print("AnthropicVertex 클라이언트 초기화 시도...")
        client = AnthropicVertex(project_id=project_id, region=location)
        print("클라이언트 초기화 성공.")

        # 3. 모델에 메시지 전송
        print("Claude 모델에 메시지 전송 시도...")
        message = client.messages.create(
            model="claude-3-5-sonnet-v2",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, Claude! If you see this message, respond with 'Connection successful.'",
                }
            ],
        )
        print("메시지 전송 성공.")

        # 4. 결과 출력
        print("\n--- 응답 결과 ---")
        print(message.content[0].text)
        print("--------------------")
        print("\n테스트 성공: Claude 모델과의 통신이 정상적으로 확인되었습니다.")

    except Exception as e:
        print("\n--- 오류 발생 ---")
        print(f"오류 유형: {type(e).__name__}")
        print(f"오류 메시지: {e}")
        print("--------------------")
        print("\n테스트 실패: Claude 모델 통신 중 오류가 발생했습니다.")

if __name__ == "__main__":
    test_claude_connection() 