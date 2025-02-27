import sys
import os
from pathlib import Path
import streamlit as st
import tiktoken
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

# 세션 상태 초기화 및 확인
def initialize_session_states():
    if "store_name" not in st.session_state:
        st.session_state.store_name = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_contents" not in st.session_state:
        st.session_state.chat_contents = []
    if "chatbot_finish" not in st.session_state:
        st.session_state.chatbot_finish = False
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

def main():
    initialize_session_states()
    # 헤더 부분
    if st.session_state.store_name:
        st.header(f"저는 :violet[*{st.session_state.store_name}*]&nbsp;&nbsp;음식점의 챗봇입니다!")
    else:
        st.subheader("Chatbot")
    
    # 메시지 컨테이너 생성
    chat_container = st.container()
    
    # 입력창을 위한 빈 공간 확보
    spacer = st.empty()
    
    # 채팅 종료 버튼과 입력창을 하단에 배치
    with st.container():
        col1, col2 = st.columns([8, 2])
        with col1:
            query = st.chat_input("질문을 입력해주세요.")
        with col2:
            if st.button("채팅 끝내기", key="end_chat"):
                st.session_state.chatbot_finish = True
                st.success("채팅이 종료되었습니다!")
                st.rerun()

    # 메시지 초기화 (환영 메시지)
    if 'messages' not in st.session_state or not st.session_state.messages:
        if st.session_state.store_name:
            welcome_message = f"안녕하세요! :violet[*{st.session_state.store_name}*]&nbsp;&nbsp;음식점에 대해 궁금한 것이 있으면 언제든 물어봐주세요!"
        else:
            welcome_message = "안녕하세요! 궁금한 것이 있으면 언제든 물어봐주세요!"
        st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
    
    # 가게 이름이 바뀌었을 때 환영 메시지 업데이트
    elif len(st.session_state.messages) > 0 and st.session_state.messages[0]["role"] == "assistant":
        if st.session_state.store_name:
            welcome_message = f"안녕하세요! :violet[*{st.session_state.store_name}*]&nbsp;&nbsp;음식점에 대해 궁금한 것이 있으면 언제든 물어봐주세요!"
        else:
            welcome_message = "안녕하세요! 궁금한 것이 있으면 언제든 물어봐주세요!"
        st.session_state.messages[0]["content"] = welcome_message

    # 메시지 표시 부분을 컨테이너 안에 배치
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat logic
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.chat_contents.append(query)

        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                if st.session_state.conversation is not None:
                    with st.spinner("Thinking..."):
                        try:
                            chain = st.session_state.conversation
                            result = chain({"question": query})
                            with get_openai_callback() as cb:
                                st.session_state.chat_history = result['chat_history']
                            response = result['answer']
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error("죄송합니다. 응답을 생성하는 중 오류가 발생했습니다.")
                            st.write(f"Error: {str(e)}")
                else:
                    st.warning("먼저 메인 페이지에서 Process를 실행해주세요!")

        # 자동 스크롤을 위한 JavaScript 실행
        js = f"""
        <script>
            function scroll() {{
                var chatElement = document.querySelector('.stChatFloatingInputContainer');
                if (chatElement) {{
                    chatElement.scrollIntoView({{behavior: 'smooth'}});
                }}
            }}
            setTimeout(scroll, 100);
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
