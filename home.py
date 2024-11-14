import streamlit as st
import tiktoken
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from utils import get_text, get_text_chunks, chunk_dataframe_to_documents, get_vectorstore, get_conversation_chain
from review_crawling import Crawling
from review_classification import Classification
import os
import pickle
import uuid
import qrcode
from io import BytesIO
from chatbot_gradio import save_chatbot_config, launch_chatbot

# OpenAI API 키 설정
openai_api_key = "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"

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


def create_chatbot(store_name, vectorstore, openai_api_key):
    """새로운 챗봇 생성 및 설정 저장"""
    # 고유 ID 생성
    store_id = str(uuid.uuid4())
    
    try:
        # 챗봇 설정 저장
        save_chatbot_config(store_id, store_name, vectorstore, openai_api_key)
        
        # Gradio 챗봇 실행 및 URL 획득
        public_url = launch_chatbot(store_id)
        return store_id, public_url
    except Exception as e:
        print(f"Error creating chatbot: {e}")
        raise

def start_processing(uploaded_files):
    with st.spinner("가게 정보를 수집 중이에요..."):
        crawler = Crawling(st.session_state.store_name)
        st.session_state.info_df = crawler.get_info()
        info_df_documents = chunk_dataframe_to_documents(st.session_state.info_df)
        
        if uploaded_files:
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            combined_chunks = text_chunks + info_df_documents
        else:
            combined_chunks = info_df_documents

        vectorstore = get_vectorstore(combined_chunks)
        # conversation chain은 저장하지 않고 vectorstore만 저장
        st.session_state.processComplete = True

    try:
        with st.spinner("챗봇을 생성하고 QR 코드를 만드는 중..."):
            # 챗봇 생성 및 URL 획득
            store_id, chatbot_url = create_chatbot(
                st.session_state.store_name,
                vectorstore,
                openai_api_key
            )
            st.write(f"{chatbot_url}")
            # 세션 상태에 저장
            st.session_state.store_id = store_id
            st.session_state.chatbot_url = chatbot_url
            
            # QR 코드 생성
            qr = qrcode.make(chatbot_url)
            img_buffer = BytesIO()
            qr.save(img_buffer, format='PNG')
            img_buffer.seek(0)
        

        # 결과 표시
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.success("가게 정보 수집이 완료되었습니다!")
            st.info(f"챗봇이 생성되었습니다! 다음 링크에서 접속하실 수 있습니다:")
            st.markdown(f"[{st.session_state.store_name} 챗봇 바로가기]({chatbot_url})")
        
        with col2:
            st.image(img_buffer, caption="챗봇 접속 QR 코드", width=200)
            
    except Exception as e:
        st.error(f"챗봇 생성 중 오류가 발생했습니다: {str(e)}")

'''
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
'''

if __name__ == "__main__":
    main()