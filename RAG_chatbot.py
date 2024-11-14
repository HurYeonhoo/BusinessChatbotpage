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

openai_api_key = "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"
fontprop = fm.FontProperties(fname='data/NanumGothic-Bold.ttf')

def main():
    st.set_page_config(
        page_title="OneClickMakerChatbot",
        
        page_icon="💬"
    )

    # CSS 
    st.markdown("""
        <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
                margin-left: -450px;  /* -50px에서 변경 */
                margin-top: -70px;  /* 위로 이동 */
            }

            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 4px;
                color: #6C7583;
                font-size: 14px;
                font-weight: 400;
                padding: 0px 0px;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: transparent;
                color: #09AB3B;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)

    # 세션 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "page" not in st.session_state:
        st.session_state.page = "main"
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 1

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
        
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(['챗봇을 생성해보아요!', '관리 페이지', 'Chatbot'])

    # 페이지 컨트롤
    if st.session_state.page == "main":
        with tab1:
            process, uploaded_files = handle_tab1_content()

        with tab2:
            handle_tab2_content()

        with tab3:
            handle_chatbot_tab(process, uploaded_files)

    if st.session_state.page == "review_analysis":
        
        with st.spinner("리뷰를 수집 중이에요..."):
            if not st.session_state.crawling_complete:  # 크롤링이 완료되지 않은 경우에만 크롤링 실행
                crawler = Crawling(st.session_state.store_name)
                out = crawler.get_reviews()
                classifica = Classification(out, openai_api_key)
                st.session_state.reviews_df = classifica.review_classification()
                st.session_state.crawling_complete = True

            if not st.session_state.review_analysis_complete:
                wc = SentimentWordCloud(st.session_state.reviews_df)
                st.session_state.response_image_pos, st.session_state.response_image_neg = wc.generate_wordcloud()
                st.session_state.review_analysis_complete = True

        st.header("리뷰 분석 결과")
        st.write(""); st.write(""); st.write("")

        col11, col22 = st.columns(2)

        # 리뷰 분석
        with col11:
            with st.expander(label="리뷰 개수 분석", expanded=True):
                st.subheader("리뷰 개수 분석")
                positive_count = st.session_state.reviews_df[st.session_state.reviews_df['label'] == 1].shape[0]
                negative_count = st.session_state.reviews_df[st.session_state.reviews_df['label'] != 1].shape[0]
                total_count = positive_count + negative_count

                st.markdown(f"""
                            총 리뷰 개수: {total_count}개

                            긍정리뷰 개수: {positive_count}개&nbsp;&nbsp;({positive_count/total_count}%)
            
                            부정리뷰 개수: {negative_count}개&nbsp;&nbsp;({negative_count/total_count}%)
                            """)             
                st.write(""); st.write(""); st.write("")


        with col22:
            with st.expander(label="리뷰 개수 시각화", expanded=True):

                st.subheader("리뷰 개수 시각화")
                review_counts = pd.DataFrame({
                    '리뷰 유형': ['긍정 리뷰', '부정 리뷰'],
                    '개수': [positive_count, negative_count]})

                fig, ax = plt.subplots()
                ax.bar(review_counts['리뷰 유형'], review_counts['개수'], color=['blue', 'orange'])
                ax.set_xlabel("리뷰 유형", fontproperties=fontprop)
                ax.set_ylabel("개수", fontproperties=fontprop)
                ax.set_title("긍정 리뷰와 부정 리뷰 개수", fontproperties=fontprop)

                ax.set_xticklabels(review_counts['리뷰 유형'], fontproperties=fontprop)
                ax.set_yticklabels(ax.get_yticks(), fontproperties=fontprop)

                st.pyplot(fig)

        col33, col44 = st.columns(2)

        with col33:
            with st.expander(label="긍정 리뷰 단어", expanded=True):
                st.image(st.session_state.response_image_pos)
                st.write("")

        with col44:
            with st.expander(label="부정 리뷰 단어", expanded=True):
                st.image(st.session_state.response_image_neg)
                st.write("")
        
        if st.button("뒤로가기"):
            st.session_state.page = "main"
            st.session_state.active_tab = 1
            st.rerun()


    elif st.session_state.page == "improvement_suggestions":

        with st.spinner("개선 방안을 분석 중이에요..."):
            # response_feedback이 없는 경우에만 새로 생성
            if st.session_state.response_feedback is None:
                feedback = ReviewFeedback(st.session_state.reviews_df, openai_api_key)
                st.session_state.response_feedback = feedback.make_feedback()
        
        st.header("개선 방안")
        st.divider()
        st.markdown(st.session_state.response_feedback)

        st.divider()
        if st.button("뒤로가기"):
            st.session_state.page = "main"
            st.session_state.active_tab = 1
            st.rerun()

    elif st.session_state.page == "marketing_tips":

        with st.spinner("마케팅 방법을 분석 중이에요..."):
            # response_marketing이 없는 경우에만 새로 생성
            if st.session_state.response_marketing is None:
                marketing = ReviewMarketing(st.session_state.reviews_df, openai_api_key)
                st.session_state.response_marketing = marketing.make_marketing()
        st.header("마케팅 방법")
        st.divider()
        st.markdown(st.session_state.response_marketing)

        st.divider()
        if st.button("뒤로가기"):
            st.session_state.page = "main"
            st.session_state.active_tab = 1
            st.rerun()

    elif st.session_state.page == "content_analysis":

        with st.spinner("대화 내용을 분석 중이에요..."):
            # 이미 분석이 완료된 경우 재분석하지 않도록 설정
            if st.session_state.response_contents is None:
                chat_analysis = ChatAnalysis(st.session_state.chat_contents, openai_api_key)
                st.session_state.response_contents = chat_analysis.make_analysis()
        
        st.header("대화 내용 분석")
        st.divider()
        st.markdown(st.session_state.response_contents)

        st.divider()
        if st.button("뒤로가기"):
            st.session_state.page = "main"
            st.session_state.active_tab = 1
            st.rerun()

    elif st.session_state.page == "store_analysis":

        st.subheader("경쟁사 가게 입력")
        name_B = st.text_input('경쟁사 가게 이름을 입력하세요!', key='name_input_B')

        if name_B and name_B != st.session_state.store_name_B:  # 가게 이름이 변경된 경우, 세션 초기화
            st.session_state.store_name_B = name_B
            st.session_state.crawling_complete_B = False
            st.session_state.info_df_B = None
            st.session_state.reviews_df_B = None

        if st.session_state.store_name_B:  # 저장된 이름이 있으면 표시
            st.markdown(f'「:violet[*{st.session_state.store_name_B}*]」 가게를 분석해드릴게요!&nbsp;start를 클릭해주세요.')

        start = st.button("start")
        st.divider()

        if start:
            st.session_state['start'] = True  # 버튼 상태를 세션에 저장
            
            if st.session_state.store_name_B: 
                with st.spinner(f'{st.session_state.store_name_B} 가게를 분석 중이에요...'):
                    # 이미 분석이 완료된 경우 재분석하지 않도록 설정
                    if st.session_state.response_store_B is None:                            
                        crawlerB = Crawling(st.session_state.store_name_B)
                        outB = crawlerB.get_reviews()
                        classificaB = Classification(outB, openai_api_key)
                        st.session_state.reviews_df_B = classificaB.review_classification()
                        sstore_analysis = StoreAnalysis(st.session_state.reviews_df, st.session_state.reviews_df_B, openai_api_key)
                        st.session_state.response_store_B = sstore_analysis.make_store_analysis()
                        st.session_state.crawling_complete_B = True
                        
            st.header("경쟁사 가게 비교")
            st.divider()
            st.markdown(st.session_state.response_store_B)
            st.divider()

        if st.button("뒤로가기"):
            st.session_state.page = "main"
            st.session_state.active_tab = 1
            st.rerun()
            
def handle_tab1_content():
    st.title(":blue[리뷰 분석] 및 :blue[챗봇 생성]💩👋")
    st.write("")
    st.markdown("""

                step☝️. 업종을 선택해주세요.
                

                step✌️. 가게 이름을 입력해주세요. 추가로 원하는 정보는 pdf를 제출하세요.
                

                step🤞. Process 버튼을 클릭하세요!

            """)
    st.divider()

    st.subheader("업종 선택")
    business_type = st.selectbox("업종을 선택하세요.", ["음식점", "미용실", "쇼핑몰", "부동산", "관광숙박업"])
    st.divider()

    st.subheader("가게 이름 입력")
    name = st.text_input('가게 이름을 입력하세요!', key='name_input')
    if name and name != st.session_state.store_name:  # 가게 이름이 변경된 경우, 세션 초기화
        st.session_state.store_name = name
        st.session_state.crawling_complete = False
        st.session_state.info_df = None
        st.session_state.reviews_df = None
        st.session_state.response_feedback = None
        st.session_state.response_marketing = None

    if st.session_state.store_name:  # 저장된 이름이 있으면 표시
        st.markdown(f'「:violet[*{st.session_state.store_name}*]」 가게 사장님 안녕하세요!')
    st.divider()

    st.subheader("PDF 제출")
    uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
    st.divider()
    
    process = st.button("Process") 

    if process:
        st.session_state['Process'] = True  # 버튼 상태를 세션에 저장
        with st.spinner("가게 정보를 수집 중이에요..."):
            start_processing(uploaded_files)  # Process 버튼 클릭 시 함수 호출

    return process, uploaded_files

def handle_tab2_content():
    if st.session_state.store_name:  # 가게 이름이 있는 경우
        st.write("")
        st.header(f":violet[*{st.session_state.store_name}*]&nbsp;&nbsp;리뷰들을 관리해보세요!")
        st.write(""); st.write(""); st.write("")
    else:  # 가게 이름이 없는 경우
        st.write("")
        st.header("리뷰들을 관리해보세요!")
        st.write(""); st.write(""); st.write("")

    col1, col2, col3 = st.columns(3)
    # 리뷰 분석
    with col1:
        with st.expander(label="리뷰 분석", expanded=True):
            st.markdown("""
                좋은 리뷰와 나쁜 리뷰를 분석해드립니다!
                
                고객의 생각을 한 눈에 확인하세요! 
                """)
            st.write("")
            if st.button("리뷰 분석"):
                st.session_state.page = "review_analysis"
                st.rerun()

    # 개선 방안
    with col2:
        with st.expander(label='개선 방안', expanded=True):
            st.markdown("""
                리뷰를 바탕으로 가게의 개선 방안을 알려드립니다!
                        
                문제점 분석을 통해 가게 성장의 새로운 기회를 제안합니다! 
                """)
            st.write("")
            if st.button("개선 방안"):
                if st.session_state.review_analysis_complete:
                    st.session_state.page = "improvement_suggestions"
                    st.rerun()
                else:
                    st.warning("리뷰 분석을 먼저 완료하세요.")

    # 마케팅 방법
    with col3:
        with st.expander(label='마케팅 추천', expanded=True):
            st.markdown("""
                고객 선호도를 반영한 맞춤형 마케팅 전략을 추천해드립니다.
                
                가게를 더욱 발전시킬 수 있어요!
                """)
            st.write("")
            if st.button("마케팅 추천"):
                if st.session_state.review_analysis_complete:
                    st.session_state.page = "marketing_tips"
                    st.rerun()
                else:
                    st.warning("리뷰 분석을 먼저 완료하세요.")

    col4, col5 = st.columns(2)

    # 동종 업계 비교 분석
    with col4:
        with st.expander(label='동종 업계 비교 분석', expanded=True):
            st.markdown("""
                다른 가게와 어떤 차이점이 있는지 비교해드려요.
                
                다른 가게와 차별화를 해보세요!
                """)
            st.write("")
            if st.button("동종 업계 비교 분석"):
                if st.session_state.review_analysis_complete:
                    st.session_state.page = "store_analysis"
                    st.rerun()
                else:
                    st.warning("리뷰 분석을 먼저 완료하세요.")


    # 대화 내용 분석
    with col5:
        with st.expander(label='대화 내용 분석', expanded=True):
            st.markdown("""
                손님이 사용한 챗봇의 대화 내용을 분석해드립니다.
                
                어떤 질문을 가장 많이 하는지 알 수 있어요!
                """)
            st.write("")
            if st.button("대화 내용 분석"):
                if st.session_state.chatbot_finish:
                    st.session_state.page = "content_analysis"
                    st.rerun()
                else:
                    st.warning("채팅이 끝나지 않았습니다.")

def start_processing(uploaded_files):
    crawler = Crawling(st.session_state.store_name)
    st.session_state.info_df = crawler.get_info()
    info_df_documents = chunk_dataframe_to_documents(st.session_state.info_df, chunk_size=900, chunk_overlap=100)
    
    # 파일이 있는 경우에만 처리
    if uploaded_files:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        combined_chunks = text_chunks + info_df_documents
    else:
        combined_chunks = info_df_documents

    # 벡터스토어 생성 및 대화 체인
    vectorstore = get_vectorstore(combined_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
    st.session_state.processComplete = True

    st.success("가게 정보 수집이 완료되었습니다!")

def handle_chatbot_tab(process, uploaded_files):
    if st.session_state.store_name:  # 가게 이름이 있는 경우
        st.write("")
        st.header(f"저는 :violet[*{st.session_state.store_name}*]&nbsp;&nbsp;음식점의 챗봇입니다!💩")
        st.write(""); st.write(""); st.write("")

    else:  # 가게 이름이 없는 경우
        st.write("")
        st.subheader("저는 챗봇입니다!💩")
        st.write(""); st.write(""); st.write("")

    if 'messages' not in st.session_state:
        if st.session_state.store_name:  # 가게 이름이 있는 경우
            welcome_message = f"안녕하세요! :violet[*{st.session_state.store_name}*]&nbsp;&nbsp;음식점에 대해 궁금한 것이 있으면 언제든 물어봐주세요!"
        else:  # 가게 이름이 없는 경우
            welcome_message = "안녕하세요! 궁금한 것이 있으면 언제든 물어봐주세요!"
        
        st.session_state['messages'] = [{"role": "assistant", "content": welcome_message}]

    # 기존 메시지를 업데이트할 때도 동일한 조건 적용
    if len(st.session_state['messages']) > 0 and st.session_state['messages'][0]["role"] == "assistant":
        if st.session_state.store_name:  # 가게 이름이 있는 경우
            welcome_message = f"안녕하세요! :violet[*{st.session_state.store_name}*]&nbsp;&nbsp; 음식점에 대해 궁금한 것이 있으면 언제든 물어봐주세요!"
        else:  # 가게 이름이 없는 경우
            welcome_message = "안녕하세요! 궁금한 것이 있으면 언제든 물어봐주세요!"
            
        st.session_state['messages'][0]["content"] = welcome_message

    for message in st.session_state.messages: 
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")  # 메모리 구현

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):  # 질문창
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.chat_contents.append(query)

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"): 
            chain = st.session_state.conversation

            with st.spinner("Thinking..."): 
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                st.markdown(response)
               
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("채팅 끝내기"):
        st.session_state.chatbot_finish = True
        
if __name__ == '__main__':
    main()