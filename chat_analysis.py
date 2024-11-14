import openai
import pandas as pd

class ChatAnalysis:
    def __init__(self, chat_contents, api_key):
        openai.api_key = api_key
        self.chat_contents = chat_contents

    def make_analysis(self):

        if self.chat_contents is None:
            raise ValueError("채팅 데이터가 제공되지 않았습니다.")

        system_prompt = f"""
        당신은 내용 분석 전문가입니다. 채팅 내용을 분석하여, 어떤 질문을 주로 하는지 분석을 제공해야합니다.

        다음은 리스트 형식의 리뷰 데이터입니다:
        {self.chat_contents}

        이 데이터를 바탕으로 다음 작업을 수행하세요:

        1. 손님들이 가장 많이 질문하는 내용을 알려주세요

        2. 주된 질문들을 알려주세요

        각 섹션에 대해 상세하고 구체적인 내용을 제공하세요. 문단과 글자 크기를 잘 출력하세요. 
        전문적이고 건설적인 톤을 유지하면서, 가게 운영자가 손님들의 질문 내용을 잘 알 수 있도록 분석해주세요.
        """
        
        return self.llm_feedback(system_prompt)

    def llm_feedback(self, system_prompt):
    
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "대화 내용 데이터를 분석하고 그에 대한 설명을 제공해주세요."}
            ],
            temperature=0,
            max_tokens=2000
        )
        response = completion.choices[0].message.content
        return response