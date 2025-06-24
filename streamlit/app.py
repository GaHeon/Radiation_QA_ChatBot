import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
GCE_SERVER_URL = os.getenv("GCE_SERVER_URL")

if not GCE_SERVER_URL:
    st.error("GCE_SERVER_URL 환경 변수를 설정해야 합니다. .env 파일을 확인하세요.")
    st.stop()

st.title("Radiation QA 챗봇")

# 대화 기록 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{GCE_SERVER_URL}/search",
                json={"query": user_input, "history": st.session_state.messages[:-1]},
                stream=True
            )
            response.raise_for_status()

            # 스트리밍 응답 표시
            response_text = st.write_stream(response.iter_content(chunk_size=None, decode_unicode=True))
        except requests.exceptions.RequestException as e:
            st.error(f"에러 발생: {e}")
            response_text = "죄송합니다. 답변을 가져오지 못했습니다."

    st.session_state.messages.append({"role": "assistant", "content": response_text})
