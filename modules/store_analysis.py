import openai
import pandas as pd

class StoreAnalysis:
    def __init__(self, df_reviews_A, df_reviews_B, api_key):
        openai.api_key = api_key
        self.df_reviews_A = df_reviews_A
        self.df_reviews_B = df_reviews_B

    def make_store_analysis(self):

        if self.df_reviews_B is None:
            raise ValueError("채팅 데이터가 제공되지 않았습니다.")

        system_prompt = f"""
        당신은 리뷰 분석 전문가입니다. 동종 업계 가게들의 리뷰를 비교해야합니다.
        사용자의 가게와 경쟁 업체 가게의 0(부정)과 1(긍정)로 구분된 리뷰 데이터를 분석하여, 해당 업체(A가게)에 대한 건설적인 피드백과 구체적인 방안을 제시해야 합니다.

        다음은 CSV 형식의 리뷰 데이터입니다. :
        사용자 가게 리뷰 데이터: {self.df_reviews_A}
        경쟁사 가게 리뷰 데이터: {self.df_reviews_B}

        이 데이터를 바탕으로 다음 작업을 수행하세요:

        1. 경쟁사 대비, 해당 업체의 긍정 리뷰, 부정 리뷰의 비율을 비교해 주세요.

        2. 해당 업체가 경쟁사 대비 어떤 점에서 차별화되는지 알려주세요.

        3. 경쟁사가 해당 업체 보다 어떤 장점을 가지는지 알려주세요.

        4. 경쟁사의 장점을 통해, 해당 업체가 어떤 점을 활용하면 좋을지 알려주세요.

        각 섹션에 대해 상세하고 구체적인 내용을 제공하세요. 문단과 글자 크기를 잘 출력하세요. 
        전문적이고 건설적인 톤을 유지하면서, 가게 운영자에게 도움이 되도록 분석을 제공해주세요.
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
