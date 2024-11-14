import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
import os
from langchain.docstore.document import Document
import tiktoken

openai_api_key = "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"

class RAGChatbot:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
        self.vector_store = None
        self.chain = None
        self.chat_history = []
    
    def init_vector_store_with_text(self, document_chunks):
        # 벡터 스토어 생성
        self.vector_store = FAISS.from_documents(document_chunks, self.embeddings)
        
        # ConversationalRetrievalChain 초기화
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0),
            retriever=self.vector_store.as_retriever(),
            memory=memory
        )
        
        return "데이터가 성공적으로 로드되었습니다. 이제 질문해주세요!"

    def chat(self, message):
        if not self.vector_store:
            return "먼저 데이터를 초기화해주세요."
        
        result = self.chain({"question": message, "chat_history": self.chat_history})
        self.chat_history.append((message, result["answer"]))
        
        return result["answer"]

'''
# Gradio 인터페이스 생성
chatbot = RAGChatbot()

# 예제 데이터 추가
example_data = """
OpenAI는 인공지능 연구소로, ChatGPT와 같은 언어 모델을 개발합니다. 
OpenAI의 목적은 인류에게 이로운 AI를 연구하고 개발하는 것입니다.
"""

# 데이터를 직접 초기화
chatbot.init_vector_store_with_text(example_data)

def respond(message, history):
    response = chatbot.chat(message)
    return response


with gr.Blocks() as demo:
    gr.Markdown("# 텍스트 기반 RAG 챗봇")
    
    chatbot_interface = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="질문을 입력하세요...", container=False),
        title="텍스트 내용에 대해 질문하세요",
    )

demo.launch()
'''
