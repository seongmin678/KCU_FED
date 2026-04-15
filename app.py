import streamlit as st
import pandas as pd
import os
import re
from fredapi import Fred

# LangChain Libraries 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. API Key Setup 
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") # 환경 변수에서 가져오기
FRED_API_KEY = "230a44e2a7c17bf323c7ad1bcbf932b7".strip() 

st.set_page_config(page_title="Fed-Watcher AI", layout="wide")
st.title("🦅 Fed-Watcher: Federal Reserve Minutes Analysis AI")

# 3.사용자 질문 분석 함수 (단위 및 날짜 추출)
def analyze_user_prompt(prompt):
    prompt_lower = prompt.lower()
    
    # 지표 및 티커 결정
    if any(keyword in prompt_lower for keyword in ['inflation', 'price', 'cpi', '물가', '인플레']):
        ticker, name, unit = "CPIAUCSL", "Inflation (CPI)", "지수 (Index, 1982-84=100)"
    elif any(keyword in prompt_lower for keyword in ['employment', 'job', 'unemployment', '고용', '실업', '일자리']):
        ticker, name, unit = "UNRATE", "Unemployment Rate", "비율 (%)"
    elif any(keyword in prompt_lower for keyword in ['gdp', 'growth', 'economy', '성장', '경기']):
        ticker, name, unit = "GDPC1", "Real GDP", "10억 달러 (Billions of $)"
    else:
        ticker, name, unit = "FEDFUNDS", "Fed Funds Rate", "금리 (%)"

    # 연도 추출 (2023-2025 같은 범위 대응)
    years = re.findall(r'\b(20\d{2})\b', prompt)
    start_year = min(years) if years else None
    end_year = max(years) if years else None
    
    return ticker, name, unit, start_year, end_year

# 4. 동적 FRED 데이터 로딩 (캐싱)
@st.cache_data
def load_dynamic_macro_data(ticker):
    try:
        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series(ticker)
        df_macro = pd.DataFrame({ticker: series})
        df_macro.index.name = 'Date'
        return df_macro
    except Exception as e:
        st.error(f"FRED API Error: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_ai_brain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory="./fed_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    template = """You are a financial expert analyzing the Federal Reserve (Fed) meeting minutes.
    Based on the documents, answer the question in English.
    {context}
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = ({"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()} 
                 | prompt | llm | StrOutputParser())
    return rag_chain

rag_chain = load_ai_brain()


# 동적 차트 렌더링 함수
def render_dynamic_chart(ticker, indicator_name, unit_desc, start_yr, end_yr):
    try:
        # 인코딩 문제 방지를 위해 utf-8-sig 사용
        df_scores = pd.read_csv("fed_scores.csv", encoding='utf-8-sig')
        df_scores['Date'] = pd.to_datetime(df_scores['Date'])
        df_scores.set_index("Date", inplace=True)
        
        df_macro = load_dynamic_macro_data(ticker)
        
        if not df_macro.empty:
            df_combined = df_scores.join(df_macro, how='outer')
            df_combined[ticker] = df_combined[ticker].ffill() 
            df_combined = df_combined.dropna(subset=['Hawkish_Score'])
            
            # X축 범위 필터링
            if start_yr:
                df_combined = df_combined[df_combined.index >= f"{start_yr}-01-01"]
            if end_yr:
                df_combined = df_combined[df_combined.index <= f"{end_yr}-12-31"]
            
            # 그래프 출력
            st.line_chart(df_combined[['Hawkish_Score', ticker]])
            
            st.markdown(f"**📈 데이터 범례 및 의미**")
            st.markdown(f"- 🔴 **Hawkish_Score**: 1~10점 (8-10: 매파/긴축, 4-7: 중립, 1-3: 비둘기파/완화)")
            st.markdown(f"- 🔵 **{indicator_name}**: {unit_desc}")
            
            if start_yr or end_yr:
                st.caption(f"📅 필터링된 기간: {start_yr if start_yr else '시작'} ~ {end_yr if end_yr else '현재'}")
            
    except Exception as e:
        st.error(f"차트를 불러오는 중 오류가 발생했습니다: {e}")


# 7. Chatbot UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "ticker" in message:
            render_dynamic_chart(message["ticker"], message["name"], message["unit"], 
                                 message["s_yr"], message["e_yr"])

# 새 질문 처리
if prompt := st.chat_input("Ask about inflation or interest rates from 2023 to 2024..."):

    t, n, u, sy, ey = analyze_user_prompt(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        col_text, col_graph = st.columns([1, 1])
        with col_text:
            with st.spinner("Analyzing Fed Minutes..."):
                real_answer = rag_chain.invoke(prompt)
                st.markdown(real_answer)
        with col_graph:
            render_dynamic_chart(t, n, u, sy, ey)
                
    st.session_state.messages.append({
        "role": "assistant", "content": real_answer, 
        "ticker": t, "name": n, "unit": u, "s_yr": sy, "e_yr": ey
    })
    
    
