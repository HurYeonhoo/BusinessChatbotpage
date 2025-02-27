import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 프로젝트 루트 디렉토리 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from review_feedback import ReviewFeedback
from review_marketing import ReviewMarketing
from review_crawling import Crawling
from review_classification import Classification
from chat_analysis import ChatAnalysis
from word import SentimentWordCloud
from store_analysis import StoreAnalysis

# 세션 상태 초기화
def initialize_session_states():
    if "page" not in st.session_state:
        st.session_state.page = "management"
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 1
    if "store_name" not in st.session_state:
        st.session_state.store_name = ""
    if "reviews_df" not in st.session_state:
        st.session_state.reviews_df = None
    if "crawling_complete" not in st.session_state:
        st.session_state.crawling_complete = False
    if "review_analysis_complete" not in st.session_state:
        st.session_state.review_analysis_complete = False
    if "response_feedback" not in st.session_state:
        st.session_state.response_feedback = None
    if "response_marketing" not in st.session_state:
        st.session_state.response_marketing = None
    if "response_image_pos" not in st.session_state:
        st.session_state.response_image_pos = None
    if "response_image_neg" not in st.session_state:
        st.session_state.response_image_neg = None
    if "chatbot_finish" not in st.session_state:
        st.session_state.chatbot_finish = False
    if "response_contents" not in st.session_state:
        st.session_state.response_contents = None
    if "chat_contents" not in st.session_state:
        st.session_state.chat_contents = []

# 폰트 설정
openai_api_key = "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"
fontprop = fm.FontProperties(fname='data/NanumGothic-Bold.ttf')

def main():
    # 페이지 시작시 세션 상태 초기화
    initialize_session_states()

    if st.session_state.page == "management":
        show_management_page()
    elif st.session_state.page == "review_analysis":
        show_review_analysis_page()
    elif st.session_state.page == "improvement_suggestions":
        show_improvement_suggestions_page()
    elif st.session_state.page == "marketing_tips":
        show_marketing_tips_page()
    elif st.session_state.page == "store_analysis":
        show_store_analysis_page()
    elif st.session_state.page == "content_analysis":
        show_content_analysis_page()

def show_management_page():
    # CSS 추가
    st.markdown("""
        <style>
            .stExpander {
                min-height: 220px !important;
            }
            .stExpander > div {
                min-height: 180px !important;
            }
            .streamlit-expanderContent {
                height: auto !important;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.store_name:  # 가게 이름이 있는 경우
        st.write("")
        st.header(f":violet[*{st.session_state.store_name}*]&nbsp;&nbsp;리뷰들을 관리해보세요!")
        st.write(""); st.write(""); st.write("")
    else:  # 가게 이름이 없는 경우
        st.write("")
        st.header("⚙️리뷰들을 관리해보세요!")
        st.write(""); st.write(""); st.write("")

    # 리뷰 분석
    with st.expander(label="📝 리뷰 분석", expanded=True):
        st.markdown("""
                고객이 남긴 좋은 리뷰와 나쁜 리뷰를 정밀하게 분석
                        
                    ✔ 긍정적인 리뷰 분석
                        
                    ✔ 부정적인 리뷰 분석

                    ✔ 주요 키워드 시각화

                """)
        if st.button("리뷰 분석"):
            if st.session_state.store_name:
                st.session_state.page = "review_analysis"
                st.rerun()
            else:
                st.warning("가게 이름을 먼저 입력하세요.")
    # 개선 방안
    with st.expander(label='🔧 개선 방안', expanded=True):
        st.markdown("""
                리뷰를 바탕으로 가게의 개선 방안 분석
                        
                    ✔ 리뷰 기반 문제점 분석

                    ✔ 서비스 및 운영 개선 제안

                """)
        if st.button("개선 방안"):
            if st.session_state.review_analysis_complete:
                st.session_state.page = "improvement_suggestions"
                st.rerun()
            else:
                st.warning("리뷰 분석을 먼저 완료하세요.")

    # 마케팅 방법
    with st.expander(label='📢 마케팅 추천', expanded=True):
        st.markdown("""
                고객의 선호도를 반영한 맞춤형 마케팅 전략 추천
                        
                    ✔ 리뷰 기반 마케팅 인사이트

                    ✔ 고객 맞춤 프로모션 제안

                """)
        if st.button("마케팅 추천"):
            if st.session_state.review_analysis_complete:
                st.session_state.page = "marketing_tips"
                st.rerun()
            else:
                st.warning("리뷰 분석을 먼저 완료하세요.")

    # 동종 업계 비교 분석
    with st.expander(label='📊 동종 업계 비교 분석', expanded=True):
        st.markdown("""
                경쟁 가게를 분석하여 차별화된 포인트 파악

                    ✔ 업계 트렌드 파악
                        
                    ✔ 경쟁 가게 비교
                        
                    ✔ 강점 및 개선점 제안
                        
                """)
        if st.button("동종 업계 비교 분석"):
            if st.session_state.review_analysis_complete:
                st.session_state.page = "store_analysis"
                st.rerun()
            else:
                st.warning("리뷰 분석을 먼저 완료하세요.")

    # 세 번째 행: 3개의 열 (빈칸-내용-빈칸 구조)
    with st.expander(label='💬 대화 내용 분석', expanded=True):
        st.markdown("""
                손님이 챗봇을 통해 남긴 대화 내용을 분석하여 고객의 관심사와 주요 문의 사항 파악
                        
                    ✔ 주요 질문 유형 분석

                    ✔ 대화 패턴 분석
                        
                """)
        if st.button("대화 내용 분석"):
            if st.session_state.chatbot_finish:
                st.session_state.page = "content_analysis"
                st.rerun()
            else:
                st.warning("채팅이 끝나지 않았습니다.")


# 각 기능별 함수 구현
def show_review_analysis_page():   
    with st.spinner("리뷰를 수집 중이에요..."):
        if not st.session_state.crawling_complete:  # 크롤링이 완료되지 않은 경우에만 크롤링 실행
            try:
                crawler = Crawling(st.session_state.store_name)
                out = crawler.get_reviews()
                classifica = Classification(out, openai_api_key)
                st.session_state.reviews_df = classifica.review_classification()
                st.session_state.crawling_complete = True
                st.session_state.review_analysis_complete = False  # 크롤링이 새로 되면 분석 상태 초기화
                st.session_state.response_image_pos = None  # 워드클라우드 초기화
                st.session_state.response_image_neg = None  # 워드클라우드 초기화
            except Exception as e:
                st.error(f"리뷰 수집 중 오류가 발생했습니다: {str(e)}")
                return

        if not st.session_state.review_analysis_complete or \
           st.session_state.response_image_pos is None or \
           st.session_state.response_image_neg is None:
            try:
                wc = SentimentWordCloud(st.session_state.reviews_df)
                st.session_state.response_image_pos, st.session_state.response_image_neg = wc.generate_wordcloud()
                st.session_state.review_analysis_complete = True
            except Exception as e:
                st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {str(e)}")
                return

    st.header("리뷰 분석")
    st.write(""); st.write(""); st.write("")

    # 분석 결과 표시
    if st.session_state.reviews_df is not None:
        col11, col22 = st.columns(2)

        # 리뷰 개수 분석
        with col11:
            with st.expander(label="리뷰 개수 분석", expanded=True):
                positive_count = st.session_state.reviews_df[st.session_state.reviews_df['label'] == 1].shape[0]
                negative_count = st.session_state.reviews_df[st.session_state.reviews_df['label'] != 1].shape[0]
                total_count = positive_count + negative_count

                st.markdown(f"""
                            총 리뷰 개수: {total_count}개

                            긍정리뷰 개수: {positive_count}개&nbsp;&nbsp;({(positive_count/total_count*100):.1f}%)
            
                            부정리뷰 개수: {negative_count}개&nbsp;&nbsp;({(negative_count/total_count*100):.1f}%)
                            """)             
                st.write(""); st.write(""); st.write("")

        # 리뷰 개수 시각화
        with col22:
            with st.expander(label="리뷰 개수 시각화", expanded=True):
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

        # 워드클라우드 표시
        if st.session_state.response_image_pos is not None and st.session_state.response_image_neg is not None:
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
        st.session_state.page = "management"
        st.rerun()


def show_improvement_suggestions_page():
    if st.session_state.reviews_df is None:
        st.error("리뷰 데이터가 없습니다. 리뷰 분석을 먼저 진행해주세요.")
        if st.button("뒤로가기"):
            st.session_state.page = "management"
            st.rerun()
        return

    with st.spinner("개선 방안을 분석 중이에요..."):
        # response_feedback이 없는 경우에만 새로 생성
        if st.session_state.response_feedback is None:
            try:
                feedback = ReviewFeedback(st.session_state.reviews_df, openai_api_key)
                st.session_state.response_feedback = feedback.make_feedback()
            except Exception as e:
                st.error("개선 방안 분석 중 오류가 발생했습니다. 리뷰 분석을 다시 진행해주세요.")
                if st.button("뒤로가기"):
                    st.session_state.page = "management"
                    st.rerun()
                return
    
    st.header("개선 방안")
    st.divider()
    st.markdown(st.session_state.response_feedback)

    st.divider()
    if st.button("뒤로가기"):
        st.session_state.page = "management"
        st.rerun()

def show_marketing_tips_page():
    if st.session_state.reviews_df is None:
        st.error("리뷰 데이터가 없습니다. 리뷰 분석을 먼저 진행해주세요.")
        if st.button("뒤로가기"):
            st.session_state.page = "management"
            st.rerun()
        return

    with st.spinner("마케팅 방법을 분석 중이에요..."):
        # response_marketing이 없는 경우에만 새로 생성
        if st.session_state.response_marketing is None:
            try:
                marketing = ReviewMarketing(st.session_state.reviews_df, openai_api_key)
                st.session_state.response_marketing = marketing.make_marketing()
            except Exception as e:
                st.error("마케팅 추천 분석 중 오류가 발생했습니다. 리뷰 분석을 다시 진행해주세요.")
                if st.button("뒤로가기"):
                    st.session_state.page = "management"
                    st.rerun()
                return

    st.header("마케팅 방법")
    st.divider()
    st.markdown(st.session_state.response_marketing)

    st.divider()
    if st.button("뒤로가기"):
        st.session_state.page = "management"
        st.rerun()

def show_store_analysis_page():
    st.subheader("경쟁사 가게 입력")
    name_B = st.text_input('경쟁사 가게 이름을 입력하세요!', key='name_input_B')

    if name_B and name_B != st.session_state.store_name_B:  # 가게 이름이 변경된 경우, 세션 초기화
        st.session_state.store_name_B = name_B
        st.session_state.crawling_complete_B = False
        st.session_state.info_df_B = None
        st.session_state.reviews_df_B = None
        st.session_state.response_store_B = None  # 새 가게 이름이 입력되면 분석 결과도 초기화

    if st.session_state.store_name_B:  # 저장된 이름이 있으면 표시
        st.markdown(f'「:violet[*{st.session_state.store_name_B}*]」 가게를 분석해드릴게요!&nbsp;start를 클릭해주세요.')

    start = st.button("start")
    st.divider()

    # 새로운 분석 시작
    if start and st.session_state.store_name_B:
        with st.spinner(f'{st.session_state.store_name_B} 가게를 분석 중이에요...'):
            if st.session_state.response_store_B is None:
                crawlerB = Crawling(st.session_state.store_name_B)
                outB = crawlerB.get_reviews()
                classificaB = Classification(outB, openai_api_key)
                st.session_state.reviews_df_B = classificaB.review_classification()
                sstore_analysis = StoreAnalysis(st.session_state.reviews_df, st.session_state.reviews_df_B, openai_api_key)
                st.session_state.response_store_B = sstore_analysis.make_store_analysis()
                st.session_state.crawling_complete_B = True

    # 이전 분석 결과가 있으면 표시
    if st.session_state.response_store_B is not None:
        st.header("경쟁사 가게 비교")
        st.divider()
        st.markdown(st.session_state.response_store_B)
        st.divider()

    if st.button("뒤로가기"):
        st.session_state.page = "management"
        st.rerun()

def show_content_analysis_page():
    # 챗봇 사용 여부 확인
    if not st.session_state.chat_contents:
        st.warning("아직 챗봇과의 대화 내용이 없습니다. 먼저 챗봇을 사용해주세요.")
        if st.button("뒤로가기"):
            st.session_state.page = "management"
            st.rerun()
        return

    # 이전 대화 길이를 저장하는 세션 상태 초기화
    if "previous_chat_length" not in st.session_state:
        st.session_state.previous_chat_length = 0

    current_chat_length = len(st.session_state.chat_contents)

    with st.spinner("대화 내용을 분석 중이에요..."):
        try:
            # 초기 분석이 없거나 대화 내용이 변경된 경우 재분석
            if (st.session_state.response_contents is None or 
                current_chat_length != st.session_state.previous_chat_length):
                
                # 대화 내용이 있는 경우에만 분석 수행
                if current_chat_length > 0:
                    chat_analysis = ChatAnalysis(st.session_state.chat_contents, openai_api_key)
                    st.session_state.response_contents = chat_analysis.make_analysis()
                    # 현재 대화 길이 저장
                    st.session_state.previous_chat_length = current_chat_length
                else:
                    st.warning("아직 대화 내용이 없습니다.")
                    st.session_state.response_contents = None
                    return
        except Exception as e:
            st.error(f"대화 내용 분석 중 오류가 발생했습니다: {str(e)}")
            if st.button("뒤로가기"):
                st.session_state.page = "management"
                st.rerun()
            return

    st.header("대화 내용 분석")
    st.divider()

    # 분석 결과가 있는 경우에만 표시
    if st.session_state.response_contents:
        st.markdown(st.session_state.response_contents)
        # 현재 대화 수 표시
        st.info(f"현재까지 총 {current_chat_length}개의 대화가 분석되었습니다.")
    else:
        st.warning("분석할 대화 내용이 없습니다. 먼저 챗봇과 대화를 나눠주세요.")

    st.divider()
    if st.button("뒤로가기"):
        st.session_state.page = "management"
        st.rerun()

if __name__ == "__main__":
    main()
