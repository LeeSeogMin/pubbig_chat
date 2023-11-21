import dotenv
dotenv.load_dotenv()

# 1. 문서 로드
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("종합.pdf")
pages = loader.load_and_split()

# 2. 문서 분할
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=tiktoken_len
)

document_splits = text_splitter.split_documents(pages)

# 3. 텍스트 임베딩
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 벡터 저장: FAISS 벡터 저장소
from langchain.vectorstores import FAISS

vcdb = FAISS.from_documents(document_splits, embedding)

# 5. 대화형 검색 및 응답 체인 설정
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

openai = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[],
    temperature=0,
    max_tokens=2000
)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=openai,
    chain_type="stuff",
    retriever=vcdb.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 5}
    ),
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
    get_chat_history=lambda h: h,
    return_source_documents=True,
    verbose=True
)

# 6. Streamlit으로 배포
import streamlit as st
import time

st.set_page_config(
page_title="공빅 AI",
page_icon=":books:")

st.title("_공공인재빅데이터융학학 :red[AI 조교]_ :books:")

# st.set_page_config(page_title="공빅 AI 조교", page_icon=":bird:")
# st.header("공빅 AI 조교 :bird:")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", 
                                    "content": "공빅 AI 조교입니다^^ 무엇을 도와드릴까요?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = conversation_chain({"question": prompt})
        response = result['answer']

        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
