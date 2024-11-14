import streamlit as st
import tiktoken
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain.docstore.document import Document

from review_feedback import ReviewFeedback
from review_marketing import ReviewMarketing
from review_crawling import Crawling
from review_classification import Classification
from chat_analysis import ChatAnalysis
from word import SentimentWordCloud
from store_analysis import StoreAnalysis
from utils import get_text, tiktoken_len, get_text_chunks, chunk_dataframe_to_documents, get_vectorstore, get_conversation_chain

# OpenAI API 키 설정
openai_api_key =  st.secrets["OPENAI_API_KEY"]

# 페이지 설정
st.set_page_config(page_title="OneClickMakerChatbot", page_icon="💬")

def initialize_session_states():
    # 세션 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "store_name" not in st.session_state:
        st.session_state.store_name = ""
    if "store_name_B" not in st.session_state:
        st.session_state.store_name_B = ""

    if "info_df" not in st.session_state:
        st.session_state.info_df = None
    if "info_df_B" not in st.session_state:
        st.session_state.info_df_B = None
    if "reviews_df" not in st.session_state:
        st.session_state.reviews_df = None
    if "reviews_df_B" not in st.session_state:
        st.session_state.reviews_df_B = None

    if "crawling_complete" not in st.session_state:
        st.session_state.crawling_complete = False
    if "crawling_complete_B" not in st.session_state:
        st.session_state.crawling_complete_B = False
    if "review_analysis_complete" not in st.session_state:
        st.session_state.review_analysis_complete = False
    if "review_analysis_complete_B" not in st.session_state:
        st.session_state.review_analysis_complete_B = False
        
    if "response_feedback" not in st.session_state:
        st.session_state.response_feedback = None
    if "response_marketing" not in st.session_state:
        st.session_state.response_marketing = None
    if "response_image_pos" not in st.session_state:
        st.session_state.response_image_pos = None
    if "response_image_neg" not in st.session_state:
        st.session_state.response_image_neg = None
    if "response_store_B" not in st.session_state:
        st.session_state.response_store_B = None

    if "chatbot_finish" not in st.session_state:
        st.session_state.chatbot_finish = False
    if "response_contents" not in st.session_state:
        st.session_state.response_contents = None
    if "chat_contents" not in st.session_state:
        st.session_state.chat_contents = []
    if "content_analysis_done" not in st.session_state:
        st.session_state.content_analysis_done = None
    if "previous_chat_length" not in st.session_state:
        st.session_state.previous_chat_length = 0
    if "process_success" not in st.session_state:
        st.session_state.process_success = False

def main():
    # 세션 상태 초기화
    initialize_session_states()

    st.title(":blue[리뷰 분석] 및 :blue[챗봇 생성]💩👋")
    st.write("")
    st.markdown("""
        step☝️. 가게 이름을 입력해주세요.
        
        step✌️. 추가로 원하는 정보는 pdf를 제출하세요.
        
        step🤞. Process 버튼을 클릭하세요!
    """)
    st.divider()

    # 업종 선택
    #st.subheader("업종 선택")
    #business_type = st.selectbox("업종을 선택하세요.", ["음식점", "미용실", "쇼핑몰", "부동산", "관광숙박업"])
    #st.divider()

    # 가게 이름 입력
    st.subheader("가게 이름 입력")
    name = st.text_input('가게 이름을 입력하세요!', key='name_input')
    # 가게 이름이 변경된 경우 모든 관련 상태 초기화
    if name and name != st.session_state.store_name:
        st.session_state.store_name = name
        st.session_state.crawling_complete = False
        st.session_state.info_df = None
        st.session_state.reviews_df = None
        st.session_state.response_feedback = None
        st.session_state.response_marketing = None
        st.session_state.processComplete = False  # Process 상태도 초기화
        st.session_state.process_success = False  # 성공 메시지 상태 초기화
        # 동종업계 비교분석 관련 상태 초기화
        st.session_state.store_name_B = ""  # 경쟁사 가게 이름 초기화
        st.session_state.crawling_complete_B = False
        st.session_state.info_df_B = None
        st.session_state.reviews_df_B = None
        st.session_state.response_store_B = None
        # 챗봇 관련 상태 초기화
        st.session_state.messages = []  # 채팅 기록 초기화
        st.session_state.conversation = None  # 대화 모델 초기화
        st.session_state.chatbot_finish = False  # 챗봇 종료 상태 초기화
        
        # 대화 분석 관련 상태 초기화
        st.session_state.chat_contents = []  # 대화 내용 초기화
        st.session_state.response_contents = None  # 분석 결과 초기화
        st.session_state.previous_chat_length = 0  # 이전 대화 길이 초기화
    
    if st.session_state.store_name:
        st.markdown(f'「:violet[*{st.session_state.store_name}*]」 가게 사장님 안녕하세요!')
    st.divider()

    # PDF 제출
    st.subheader("PDF 제출")
    uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
    st.divider()

    process = st.button("Process")

    # Process 버튼이 클릭되었을 때
    if process:
        with st.spinner("가게 정보를 수집 중이에요..."):
            start_processing(uploaded_files)  # Process 실행
            st.session_state.process_success = True  # 성공 상태 저장

    # 성공 메시지 표시 (Process 완료 상태가 유지되는 동안)
    if st.session_state.get('process_success', False):
        st.success("가게 정보 수집이 완료되었습니다!")

def start_processing(uploaded_files):
    crawler = Crawling(st.session_state.store_name)
    st.session_state.info_df = crawler.get_info()
    info_df_documents = chunk_dataframe_to_documents(st.session_state.info_df, chunk_size=900, chunk_overlap=100)
    
    if uploaded_files:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        combined_chunks = text_chunks + info_df_documents
    else:
        combined_chunks = info_df_documents

    vectorstore = get_vectorstore(combined_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
    st.session_state.processComplete = True


if __name__ == "__main__":
    main()
