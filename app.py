import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os


# 환경변수 로드
load_dotenv()
OPEN_API_KEY = os.environ.get("OPENAI_API_KEY")


# 페이지 설정
st.set_page_config(
    page_title="F&F 인턴십 가이드 챗봇",
    #page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
    }
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.3rem;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border-radius: 0.3rem;
        cursor: pointer;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
    """
    RAG 시스템 로드 (캐싱)
    """
    try:
        # 임베딩 모델
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPEN_API_KEY
        )
        
        # 벡터 DB 로드
        db_path = "./chroma_db"
        if not Path(db_path).exists():
            st.error("❌ 벡터 DB를 찾을 수 없습니다. 먼저 main.py를 실행하세요!")
            st.stop()
        
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        # LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPEN_API_KEY
        )
        
        # Retriever
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # RAG 체인
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain, vectordb
    
    except Exception as e:
        st.error(f"❌ RAG 시스템 로드 실패: {e}")
        st.stop()


def display_message(role, content, sources=None):
    """
    메시지 표시
    """
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 당신:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 AI:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            source_text = "📚 **출처:**\n\n"
            for i, doc in enumerate(sources, 1):
                page = doc.metadata['page_num']
                title = doc.metadata['title']
                category = doc.metadata.get('category', '기타')
                source_text += f"{i}. 📄 **페이지 {page}**: {title} (카테고리: {category})\n\n"
            
            st.markdown(f"""
            <div class="source-box">
                {source_text}
            </div>
            """, unsafe_allow_html=True)


def main():
    """
    메인 앱
    """
    # 헤더
    st.markdown('<h1 class="main-header">🤖 F&F 인턴십 가이드 챗봇</h1>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=F%26F+Chatbot", use_container_width=True)
        st.markdown("### 📖 사용 가이드")
        st.markdown("""
        이 챗봇은 **F&F 인턴십 가이드**를 학습했습니다.
        
        **질문 예시:**
        - 인턴십 평가 기준이 뭐야?
        - F&F가 운영하는 브랜드는?
        - 휴가는 어떻게 신청해?
        - 조직 문화는 어때?
        - 복리후생에는 뭐가 있어?
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ 설정")
        
        # 대화 초기화 버튼
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 통계")
        
        # RAG 시스템 로드
        qa_chain, vectordb = load_rag_system()
        
        doc_count = vectordb._collection.count()
        st.metric("저장된 문서", f"{doc_count}개")
        
        if "messages" in st.session_state:
            st.metric("대화 수", len(st.session_state.messages) // 2)
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 환영 메시지
    if len(st.session_state.messages) == 0:
        st.info("💬 F&F 인턴십 가이드에 대해 무엇이든 물어보세요!")
    
    # 대화 히스토리 표시
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("sources")
        )
    
    # 질문 예시 버튼
    st.markdown("### 💡 빠른 질문")
    col1, col2, col3 = st.columns(3)
    
    example_questions = [
        "인턴십 평가 기준이 뭐야?",
        "F&F가 운영하는 브랜드는?",
        "휴가는 어떻게 신청해?"
    ]
    
    for col, question in zip([col1, col2, col3], example_questions):
        with col:
            if st.button(question, key=question):
                st.session_state.current_question = question
                st.rerun()
    
    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요...")
    
    # 예시 질문 클릭 처리
    if "current_question" in st.session_state:
        user_input = st.session_state.current_question
        del st.session_state.current_question
    
    # 질문 처리
    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # 로딩 표시
        with st.spinner("🤔 답변 생성 중..."):
            try:
                # RAG 체인 실행
                result = qa_chain({"query": user_input})
                answer = result["result"]
                sources = result["source_documents"]
                
                # AI 메시지 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"⚠️ 오류 발생: {e}")
        
        # 리프레시
        st.rerun()


if __name__ == "__main__":
    main()