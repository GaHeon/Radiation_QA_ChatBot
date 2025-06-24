import requests
import json

url = "http://127.0.0.1:8000/search"
data = {
    "query": "방사선 장비의 안전 점검은 어떻게 하나요?",
    "history": []
}

headers = {
    "Content-Type": "application/json"
}

print("요청을 보냅니다...")
try:
    with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
        print(f"서버 응답 코드: {r.status_code}")
        r.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        
        content_received = False
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                content_received = True
                print(chunk, end='', flush=True)
        
        if not content_received:
            print("응답 내용이 없습니다.")

    print()
except requests.exceptions.RequestException as e:
    print(f"오류 발생: {e}")
finally:
    print("클라이언트 실행 종료.") 