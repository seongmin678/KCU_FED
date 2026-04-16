import streamlit as st
import pandas as pd
import os
import re
import random
from fredapi import Fred
from dotenv import load_dotenv
import plotly.graph_objects as go

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "230a44e2a7c17bf323c7ad1bcbf932b7").strip()

st.set_page_config(page_title="FED Data Analyzer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
.stApp { background-color: #0D1B2A; font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background-color: #112236; border-right: 1px solid #1E3A5F; }
[data-testid="stSidebar"] * { color: #C8D8E8 !important; }
.main .block-container { padding: 1.5rem 2rem; max-width: 100%; }
.fed-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem; color: #E8F0F8; letter-spacing: 0.02em; margin-bottom: 0.2rem; }
.fed-subtitle { font-size: 0.78rem; color: #6B8FAF; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1.5rem; }
.section-label { font-size: 0.75rem; color: #6B8FAF; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.6rem; }
.chat-bubble-user { background: #1E3A5F; border: 1px solid #2A5080; border-radius: 12px 12px 4px 12px; padding: 0.8rem 1.1rem; color: #E8F0F8; font-size: 0.92rem; margin-bottom: 0.5rem; max-width: 88%; margin-left: auto; }
.chat-bubble-ai { background: #162A3F; border: 1px solid #1E3A5F; border-radius: 4px 12px 12px 12px; padding: 0.8rem 1.1rem; color: #C8D8E8; font-size: 0.92rem; margin-bottom: 0.3rem; }
.source-tag { display: inline-block; background: #1A3350; border: 1px solid #2A5070; border-radius: 6px; padding: 2px 9px; font-size: 0.74rem; color: #7BAFD4; margin: 3px 3px 3px 0; }
.history-item { background: #162A3F; border: 1px solid #1E3A5F; border-radius: 8px; padding: 0.45rem 0.8rem; color: #9BB8D4; font-size: 0.81rem; margin-bottom: 5px; }
.empty-chart { background: #112236; border: 1px dashed #1E3A5F; border-radius: 12px; padding: 3.5rem 1rem; text-align: center; color: #3A6080; font-size: 0.88rem; }
.stButton > button { background: #162A3F !important; border: 1px solid #2A5080 !important; color: #9BB8D4 !important; border-radius: 8px !important; font-size: 0.82rem !important; padding: 0.4rem 0.8rem !important; width: 100%; text-align: left !important; transition: all 0.2s !important; }
.stButton > button:hover { background: #1E3A5F !important; color: #E8F0F8 !important; border-color: #4A90C4 !important; }
div[data-testid="stHorizontalBlock"] .stButton > button { width: auto !important; padding: 0.3rem 1rem !important; background: #1A3A6A !important; border: 1px solid #4A90C4 !important; color: #7BAFD4 !important; font-size: 0.78rem !important; letter-spacing: 0.05em !important; }
div[data-testid="stHorizontalBlock"] .stButton > button:hover { background: #4A90C4 !important; color: #E8F0F8 !important; }
p, li, span { color: #C8D8E8; }
h1, h2, h3 { color: #E8F0F8 !important; }
hr { border-color: #1E3A5F; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

INDICATORS = {
    "FEDFUNDS": "기준금리 (Fed Funds Rate)",
    "DGS10": "10년물 국채금리",
    "DGS2": "2년물 국채금리",
    "T10Y2Y": "장단기 금리차 (10Y-2Y)",
    "CPIAUCSL": "소비자물가지수 (CPI)",
    "PCEPI": "PCE 물가지수",
    "UNRATE": "실업률",
    "PAYEMS": "비농업 취업자수",
    "GDPC1": "실질 GDP",
    "M2SL": "M2 통화량",
}

KEYWORD_MAP = {
    ("inflation", "price", "cpi", "물가", "인플레"): ("CPIAUCSL", "지수"),
    ("pce",): ("PCEPI", "지수"),
    ("unemployment", "employment", "job", "고용", "실업", "일자리"): ("UNRATE", "비율 (%)"),
    ("payroll", "취업자", "고용자"): ("PAYEMS", "천 명"),
    ("gdp", "growth", "economy", "성장", "경기"): ("GDPC1", "10억 달러"),
    ("10년", "10-year", "dgs10", "장기금리"): ("DGS10", "금리 (%)"),
    ("2년", "2-year", "dgs2", "단기금리"): ("DGS2", "금리 (%)"),
    ("금리차", "yield curve", "t10y2y", "장단기"): ("T10Y2Y", "금리차 (%)"),
    ("m2", "통화량", "money supply"): ("M2SL", "십억 달러"),
    ("rate", "금리", "interest", "fed funds", "기준금리"): ("FEDFUNDS", "금리 (%)"),
}

EXAMPLE_QUESTIONS = [
    "최근 5년간 금리 변동 알려줘",
    "금리 인상기의 실업률 변화는?",
    "GDP 성장률 비교해줘",
    "2022년 인플레이션 상황은?",
    "파월이 최근에 뭐라고 했어?",
    "연준이 금리를 올린 이유가 뭐야?",
]

LOADING_MESSAGES = [
    "📡 연준 문서 분석 중...",
    "🔍 회의록에서 관련 내용 검색 중...",
    "🦅 파월 의장의 발언을 찾는 중...",
    "📊 FRED 데이터와 연결 중...",
    "🏦 연준 아카이브 탐색 중...",
]

CHART_TYPES = {"Line": "line", "Bar": "bar", "Scatter": "scatter", "Area": "area"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0D1B2A",
    font=dict(color="#C8D8E8", family="DM Sans"),
    xaxis=dict(
        showgrid=True, gridcolor="#1E3A5F", gridwidth=0.5,
        tickfont=dict(color="#6B8FAF", size=11),
        tickformat="%Y", dtick="M12",
        showline=False, zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#1E3A5F", gridwidth=0.5,
        tickfont=dict(color="#6B8FAF", size=11),
        showline=False, zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(17,34,54,0.8)",
        bordercolor="#1E3A5F", borderwidth=1,
        font=dict(color="#C8D8E8", size=11),
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#162A3F", bordercolor="#2A5080", font=dict(color="#E8F0F8")),
)

def analyze_prompt(prompt: str):
    lower = prompt.lower()
    tickers = []
    for keywords, (ticker, unit) in KEYWORD_MAP.items():
        if any(k in lower for k in keywords):
            tickers.append((ticker, INDICATORS.get(ticker, ticker), unit))
    if not tickers:
        tickers = [("FEDFUNDS", "기준금리 (Fed Funds Rate)", "금리 (%)")]
    years = re.findall(r'\b(20\d{2})\b', prompt)
    start_year = min(years) if years else None
    end_year = max(years) if years else None
    return tickers, start_year, end_year

@st.cache_data(show_spinner=False)
def load_fred_data(ticker: str):
    try:
        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series(ticker)
        df = pd.DataFrame({ticker: series})
        df.index.name = "Date"
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    db = Chroma(persist_directory="./fed_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    template = """You are a senior Federal Reserve analyst with deep expertise in monetary policy.
Answer the question based on the provided Fed documents (meeting minutes, speeches).

Rules:
- If the question is in Korean, answer in Korean. If in English, answer in English.
- Be specific and cite dates/meetings when relevant.
- Keep answers concise but informative (3-5 sentences).
- If the documents don't contain enough info, say so honestly.

Context from Fed documents:
{context}

Question: {question}

Answer:"""
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
        | prompt_template | llm | StrOutputParser()
    )
    return chain, retriever

def get_sources(retriever, question: str):
    try:
        docs = retriever.invoke(question)
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "")
            if src:
                basename = os.path.basename(src).replace(".txt", "").replace("_", " ")
                if basename not in sources:
                    sources.append(basename)
        return sources[:3]
    except:
        return []

def make_trace(chart_type, x, y, name, color, secondary=False):
    opacity = 0.7 if secondary else 1.0
    if chart_type == "bar":
        return go.Bar(x=x, y=y, name=name, marker_color=color, opacity=opacity)
    elif chart_type == "scatter":
        return go.Scatter(x=x, y=y, name=name, mode="markers",
                         marker=dict(color=color, size=5, opacity=opacity))
    elif chart_type == "area":
        return go.Scatter(x=x, y=y, name=name, mode="lines",
                         fill="tozeroy", line=dict(color=color, width=2),
                         fillcolor=color.replace(")", f",{0.15})").replace("rgb", "rgba") if "rgb" in color else color,
                         opacity=opacity)
    else:  # line (default)
        return go.Scatter(x=x, y=y, name=name, mode="lines",
                         line=dict(color=color, width=2), opacity=opacity)

def render_chart(tickers, start_yr, end_yr, chart_type="line"):
    try:
        df_scores = pd.read_csv("fed_scores.csv", encoding="utf-8-sig")
        df_scores["Date"] = pd.to_datetime(df_scores["Date"])
        df_scores.set_index("Date", inplace=True)

        colors_main = ["#4A90C4", "#5BC4A0", "#F0A050", "#E06080", "#8B70D8"]
        hawkish_color = "#E05050"

        for idx, (ticker, name, unit) in enumerate(tickers):
            df_macro = load_fred_data(ticker)
            if df_macro.empty:
                continue
            df_combined = df_scores.join(df_macro, how="outer")
            df_combined[ticker] = df_combined[ticker].ffill()
            df_combined = df_combined.dropna(subset=["Hawkish_Score"])
            if start_yr:
                df_combined = df_combined[df_combined.index >= f"{start_yr}-01-01"]
            if end_yr:
                df_combined = df_combined[df_combined.index <= f"{end_yr}-12-31"]

            fig = go.Figure()
            color = colors_main[idx % len(colors_main)]

            # 지표 트레이스 (왼쪽 y축)
            fig.add_trace(make_trace(chart_type, df_combined.index, df_combined[ticker], name, color))

            # Hawkish Score 트레이스 (오른쪽 y축)
            hawkish_trace = make_trace(chart_type, df_combined.index, df_combined["Hawkish_Score"],
                                       "Hawkish Score", hawkish_color, secondary=True)
            hawkish_trace.update(yaxis="y2")
            fig.add_trace(hawkish_trace)

            layout = PLOTLY_LAYOUT.copy()
            layout.update(
                title=dict(text=f"{name} <span style='font-size:12px;color:#6B8FAF'>({unit})</span>",
                          font=dict(color="#E8F0F8", size=14), x=0),
                yaxis2=dict(
                    title="Hawkish Score",
                    overlaying="y", side="right",
                    showgrid=False,
                    tickfont=dict(color="#6B8FAF", size=11),
                    range=[0, 12],
                    zeroline=False,
                ),
                height=260,
            )
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except Exception as e:
        st.error(f"차트 오류: {e}")

# ── 세션 초기화 ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "line"

rag_chain, retriever = load_rag()

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="fed-title">FED Data Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="fed-subtitle">Federal Reserve Intelligence</div>', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    if st.session_state.history:
        st.markdown('<p class="section-label">History</p>', unsafe_allow_html=True)
        for q in reversed(st.session_state.history[-10:]):
            short = q[:38] + "..." if len(q) > 38 else q
            st.markdown(f'<div class="history-item">💬 {short}</div>', unsafe_allow_html=True)
        st.markdown("---")
    st.markdown('<p class="section-label">사용 가능한 지표</p>', unsafe_allow_html=True)
    for ticker, name in INDICATORS.items():
        st.markdown(
            f"<div style='font-size:0.79rem;margin-bottom:5px;'>"
            f"<code style='background:#1A3350;color:#7BAFD4;padding:1px 5px;border-radius:4px;font-size:0.73rem;'>{ticker}</code>"
            f" <span style='color:#9BB8D4;'>{name}</span></div>",
            unsafe_allow_html=True
        )

# ── 메인 레이아웃 ─────────────────────────────────────────────
col_chart, col_chat = st.columns([1.1, 1], gap="large")

with col_chart:
    st.markdown('<p class="section-label">Chart</p>', unsafe_allow_html=True)

    # 차트 타입 선택 버튼
    last_ai = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    if last_ai and "tickers" in last_ai:
        btn_cols = st.columns(len(CHART_TYPES))
        for i, (label, ctype) in enumerate(CHART_TYPES.items()):
            with btn_cols[i]:
                is_active = st.session_state.chart_type == ctype
                if st.button(label, key=f"chart_btn_{ctype}",
                             help=f"{label} 차트로 보기"):
                    st.session_state.chart_type = ctype
                    st.rerun()

        render_chart(last_ai["tickers"], last_ai.get("start_yr"), last_ai.get("end_yr"),
                     st.session_state.chart_type)

        st.markdown('<p class="section-label" style="margin-top:0.5rem;">사용한 지표</p>', unsafe_allow_html=True)
        for ticker, name, unit in last_ai["tickers"]:
            st.markdown(f'<span class="source-tag">{ticker}: {name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-chart">질문을 입력하면<br>관련 차트가 여기에 표시됩니다</div>', unsafe_allow_html=True)

with col_chat:
    st.markdown('<p class="section-label">Analysis</p>', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown("<p style='font-size:0.83rem;color:#6B8FAF;margin-bottom:0.5rem;'>예시 질문</p>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, eq in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(eq, key=f"eq_{i}"):
                    st.session_state["pending_question"] = eq
                    st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                src_html = "".join(f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"])
                st.markdown(f"<div style='margin-bottom:0.8rem;'>{src_html}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("질문을 입력하세요. 예: 최근 5년 금리 그래프 보여줘")
    if "pending_question" in st.session_state:
        user_input = st.session_state.pop("pending_question")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.history.append(user_input)
        tickers, start_yr, end_yr = analyze_prompt(user_input)
        with st.spinner(random.choice(LOADING_MESSAGES)):
            answer = rag_chain.invoke(user_input)
            sources = get_sources(retriever, user_input)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "tickers": tickers,
            "start_yr": start_yr,
            "end_yr": end_yr,
        })
        st.rerun()
