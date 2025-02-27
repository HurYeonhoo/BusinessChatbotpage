import pandas as pd
import re
from konlpy.tag import Okt
from collections import Counter
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO

class SentimentWordCloud:
    def __init__(self, review_labels):
        self.review_labels = review_labels

    def generate_wordcloud(self):
        # 불용어 다운로드
        url = "https://raw.githubusercontent.com/byungjooyoo/Dataset/main/korean_stopwords.txt"
        response = requests.get(url)
        stop_words = response.text.split("\n")
        
        # 폰트 설정
        mpl.font_manager.fontManager.addfont('data/NanumGothic-Bold.ttf')

        positive_data = self.review_labels[self.review_labels['label'] == 1].copy()
        negative_data = self.review_labels[self.review_labels['label'] != 1].copy()
        
        # 필요 없는 문자 제거
        pattern = re.compile(r'[가-힣\s]+')
        positive_data['ko_text'] = positive_data['comment'].apply(lambda text: ' '.join(pattern.findall(text)))
        negative_data['ko_text'] = negative_data['comment'].apply(lambda text: ' '.join(pattern.findall(text)))

        # 문자열 하나로 병합
        positive_ko_text = ','.join(positive_data['ko_text'].dropna())
        negative_ko_text = ','.join(negative_data['ko_text'].dropna())

        # 형태소 분류
        okt = Okt()
        positive_nouns = [n for n in okt.nouns(positive_ko_text) if len(n) > 1 and n not in stop_words]
        negative_nouns = [n for n in okt.nouns(negative_ko_text) if len(n) > 1 and n not in stop_words]

        # 명사 추출
        positive_count = Counter(positive_nouns).most_common(30)
        negative_count = Counter(negative_nouns).most_common(30)
        
        # 색상을 긍정/부정에 따라 지정하는 함수
        def color_func_positive(word, *args, **kwargs):
            return 'blue'

        def color_func_negative(word, *args, **kwargs):
            return 'red'

        # 긍정 워드 클라우드 생성
        wc_positive = WordCloud(
            font_path='data/NanumGothic-Bold.ttf',
            background_color='white',
            width=800,
            height=600,
            color_func=color_func_positive
        ).generate_from_frequencies(dict(positive_count))

        # 부정 워드 클라우드 생성
        wc_negative = WordCloud(
            font_path='data/NanumGothic-Bold.ttf',
            background_color='white',
            width=800,
            height=600,
            color_func=color_func_negative
        ).generate_from_frequencies(dict(negative_count))

        # 각각의 워드 클라우드를 이미지로 저장하고 반환
        img_buffer_positive = BytesIO()
        img_buffer_negative = BytesIO()

        # 긍정 워드 클라우드 저장
        plt.figure(figsize=(10, 10))
        plt.imshow(wc_positive, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img_buffer_positive, format='png', dpi=300, bbox_inches='tight', transparent=True)
        img_buffer_positive.seek(0)

        # 부정 워드 클라우드 저장
        plt.figure(figsize=(10, 10))
        plt.imshow(wc_negative, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img_buffer_negative, format='png', dpi=300, bbox_inches='tight', transparent=True)
        img_buffer_negative.seek(0)

        return img_buffer_positive.getvalue(), img_buffer_negative.getvalue()
