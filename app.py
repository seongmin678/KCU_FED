import os
import sys

# SQLite3 override for ChromaDB (Required for Render/Linux environments with old sqlite3)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Disable Chroma Telemetry to prevent hanging
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import re
import datetime
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from fredapi import Fred
import ssl

# Fix macOS Python 3 SSL Certificate error for fredapi
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv
import plotly.graph_objects as go
import json

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from apscheduler.schedulers.background import BackgroundScheduler

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()

# Flask 앱 초기화 및 정적 파일 경로 지정 (현재 디렉토리 기준)
app = Flask(__name__, static_folder=".")
CORS(app)

# ── SEP 데이터 로드 ──
SEP_CONTEXT = ""
SEP_LATEST_DATA = {}
try:
    if os.path.exists("sep_values.csv"):
        df_sep = pd.read_csv("sep_values.csv")
        if not df_sep.empty:
            latest_row = df_sep.iloc[-1]
            SEP_LATEST_DATA = latest_row.to_dict()
            date = str(latest_row['Date'])
            SEP_CONTEXT = f"--- LATEST FED SEP PROJECTIONS (As of {date}) ---\n"
            SEP_CONTEXT += f"GDP Growth: Year 1: {latest_row.get('GDP_Year1')}%, Year 2: {latest_row.get('GDP_Year2')}%, Year 3: {latest_row.get('GDP_Year3')}%, Longer Run: {latest_row.get('GDP_LongerRun')}%\n"
            SEP_CONTEXT += f"Unemployment Rate: Year 1: {latest_row.get('UNRATE_Year1')}%, Year 2: {latest_row.get('UNRATE_Year2')}%, Year 3: {latest_row.get('UNRATE_Year3')}%, Longer Run: {latest_row.get('UNRATE_LongerRun')}%\n"
            SEP_CONTEXT += f"PCE Inflation: Year 1: {latest_row.get('PCE_Year1')}%, Year 2: {latest_row.get('PCE_Year2')}%, Year 3: {latest_row.get('PCE_Year3')}%, Longer Run: {latest_row.get('PCE_LongerRun')}%\n"
            SEP_CONTEXT += f"Core PCE Inflation: Year 1: {latest_row.get('CORE_PCE_Year1')}%, Year 2: {latest_row.get('CORE_PCE_Year2')}%, Year 3: {latest_row.get('CORE_PCE_Year3')}%\n"
            SEP_CONTEXT += f"Fed Funds Rate: Year 1: {latest_row.get('FEDFUNDS_Year1')}%, Year 2: {latest_row.get('FEDFUNDS_Year2')}%, Year 3: {latest_row.get('FEDFUNDS_Year3')}%, Longer Run: {latest_row.get('FEDFUNDS_LongerRun')}%\n"
            SEP_CONTEXT += "-"*40
except Exception as e:
    print(f"Error loading SEP data: {e}")

# ── 1. 스케줄러: 연준 문서 크롤링 ────────────────────────────────────
def update_vector_db():
    print(f"[{datetime.datetime.now()}] 신규 연설문/회의록 수집 & ChromaDB 업데이트 실행 중...")
    try:
        import requests
        from bs4 import BeautifulSoup
        import xml.etree.ElementTree as ET
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.docstore.document import Document
        
        rss_url = "https://www.federalreserve.gov/feeds/press_monetary.xml"
        response = requests.get(rss_url)
        root = ET.fromstring(response.content)
        
        urls_to_scrape = []
        for item in root.findall('./channel/item')[:3]:
            link = item.find('link').text
            if link:
                urls_to_scrape.append(link)
                
        docs = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        for url in urls_to_scrape:
            page_resp = requests.get(url, headers=headers)
            soup = BeautifulSoup(page_resp.text, 'html.parser')
            article = soup.find('div', id='article')
            if not article:
                article = soup.find('body')
            text = article.get_text(separator='\n', strip=True) if article else ""
            if text:
                docs.append(Document(page_content=text, metadata={"source": url}))
                
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", max_retries=0)
            db = Chroma(persist_directory="./fed_db", embedding_function=embeddings)
            db.add_documents(split_docs)
            print(f"[{datetime.datetime.now()}] 업데이트 완료: {len(split_docs)}개 청크 추가됨.")
        else:
            print(f"[{datetime.datetime.now()}] 수집할 문서가 없습니다.")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] 업데이트 중 오류 발생: {e}")

# 스케줄러를 한 번만 실행되도록 설정
scheduler = BackgroundScheduler(timezone="Asia/Seoul")
scheduler.add_job(update_vector_db, 'cron', hour=4, minute=0)
scheduler.start()

# ── 2. 지표 설정 및 분할기 ──────────────────────────────────────────
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
    ("inflation", "price", "cpi", "물가", "인플레"): ("CPIAUCSL", "지수 (Index, 1982-84=100)"),
    ("pce",): ("PCEPI", "지수"),
    ("unemployment", "employment", "job", "고용", "실업", "일자리"): ("UNRATE", "비율 (%)"),
    ("payroll", "취업자", "고용자"): ("PAYEMS", "천 명"),
    ("gdp", "growth", "economy", "성장", "경기"): ("GDPC1", "10억 달러 (Billions of $)"),
    ("10년", "10-year", "dgs10", "장기금리"): ("DGS10", "금리 (%)"),
    ("2년", "2-year", "dgs2", "단기금리"): ("DGS2", "금리 (%)"),
    ("금리차", "yield curve", "t10y2y", "장단기"): ("T10Y2Y", "금리차 (%)"),
    ("m2", "통화량", "money supply"): ("M2SL", "십억 달러"),
    ("rate", "금리", "interest", "fed funds", "기준금리"): ("FEDFUNDS", "금리 (%)"),
}

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
    
    recent_years = re.search(r'최근\s*(\d+)년', prompt)
    if recent_years and not start_year:
        n_years = int(recent_years.group(1))
        start_year = str(datetime.datetime.now().year - n_years)

    is_relation = any(w in lower for w in ["relationship", "관계", "상관", "비교"])
    chart_type = "scatter_xy" if is_relation and len(tickers) == 2 else "line"
    return tickers, start_year, end_year, chart_type

def load_fred_data(ticker: str):
    try:
        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series(ticker)
        df = pd.DataFrame({ticker: series})
        df.index.name = "Date"
        return df
    except Exception as e:
        return pd.DataFrame()

# ── 3. Langchain RAG 초기화 ──────────────────────────────────────────
_rag_chain = None
_retriever = None

def get_rag_chain_and_retriever():
    global _rag_chain, _retriever
    if _rag_chain is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", max_retries=0)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_retries=0)
        db = Chroma(persist_directory="./fed_db", embedding_function=embeddings)
        _retriever = db.as_retriever(search_kwargs={"k": 4})

        template = f"""You are a senior Federal Reserve analyst with deep expertise in monetary policy.
Answer the question based on the provided Fed documents (meeting minutes, speeches) and the latest SEP projections.

Rules:
- If the question is in Korean, answer in Korean. If in English, answer in English.
- Be specific and cite numbers and dates/meetings when relevant.
- Keep answers concise but informative (3-5 sentences).
- If the documents don't contain enough info, say so honestly.
- NEVER apologize for not being able to show graphs or charts. The system automatically renders graphs in the user interface based on the user's keywords. Just analyze the requested topic based on text data as if the chart is naturally provided below.

Latest SEP Forward Projections:
{SEP_CONTEXT}

Context from Fed documents:
{{context}}

Question: {{question}}

Answer:"""
        prompt_template = ChatPromptTemplate.from_template(template)
        _rag_chain = (
            {"context": _retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
            | prompt_template | llm | StrOutputParser()
        )
    return _rag_chain, _retriever

def get_sources(question: str):
    try:
        _, retriever = get_rag_chain_and_retriever()
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

# ── 4. 라우팅 ──────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.json
        question = data.get("message", "")
        if not question:
            return jsonify({"error": "No message provided"}), 400

        tickers, start_yr, end_yr, chart_type = analyze_prompt(question)
        rag_chain, _ = get_rag_chain_and_retriever()
        answer = rag_chain.invoke(question)
        sources = get_sources(question)

        return jsonify({
            "answer": answer,
            "sources": sources,
            "tickers": tickers,
            "start_yr": start_yr,
            "end_yr": end_yr,
            "chart_type": chart_type
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"서버 내부 오류: {str(e)}"}), 500

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
    else:  # line
        return go.Scatter(x=x, y=y, name=name, mode="lines",
                         line=dict(color=color, width=2), opacity=opacity)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFBF0",
    plot_bgcolor="#FFFBF0",
    font=dict(color="#1f2937", family="Pretendard"),
    xaxis=dict(
        title="연도 (Year)",
        showgrid=True, gridcolor="#9ca3af", gridwidth=1,
        tickfont=dict(color="#4b5563", size=11),
        tickformat="%Y",
        showline=True, linecolor="#6b7280", linewidth=1, zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#9ca3af", gridwidth=1,
        tickfont=dict(color="#4b5563", size=11),
        showline=True, linecolor="#6b7280", linewidth=1, zeroline=False,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.05,
        xanchor="right", x=1,
        bgcolor="rgba(255,251,240,0.8)",
        bordercolor="#cbd5e1", borderwidth=1,
        font=dict(color="#1f2937", size=11),
    ),
    margin=dict(l=50, r=20, t=60, b=50),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#3b82f6", font=dict(color="#1f2937")),
)

@app.route("/api/chart", methods=["POST"])
def api_chart():
    data = request.json
    tickers = data.get("tickers", [])
    start_yr = data.get("start_yr")
    end_yr = data.get("end_yr")
    chart_type = data.get("chart_type", "line")

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    try:
        colors_main = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

        if chart_type == "scatter_xy" and len(tickers) == 2:
            t1, name1, unit1 = tickers[0]
            t2, name2, unit2 = tickers[1]
            df1 = load_fred_data(t1)
            df2 = load_fred_data(t2)
            if df1.empty or df2.empty:
                return jsonify({"error": "No valid data to plot"}), 404
            
            df_combined = df1.join(df2, how="inner").dropna()
            if start_yr: df_combined = df_combined[df_combined.index >= f"{start_yr}-01-01"]
            if end_yr: df_combined = df_combined[df_combined.index <= f"{end_yr}-12-31"]
            
            if df_combined.empty:
                return jsonify({"error": "No valid data to plot"}), 404
            
            fig = go.Figure(data=go.Scatter(
                x=df_combined[t1].tolist(), y=df_combined[t2].tolist(), mode='markers',
                marker=dict(size=6, color="#3b82f6", opacity=0.6),
                name="Relationship"
            ))
            layout = PLOTLY_LAYOUT.copy()
            layout.update(
                title=dict(text=f"{name1} vs {name2}", font=dict(color="#1f2937", size=12), x=0),
                xaxis=dict(title=dict(text=f"{name1} ({unit1})", font=dict(size=12, color="#4b5563")), tickformat=""),
                yaxis=dict(title=dict(text=f"{name2} ({unit2})", font=dict(size=12, color="#4b5563"))),
                hovermode="closest",
                height=320,
            )
            fig.update_layout(**layout)
            return fig.to_json()

        fig = go.Figure()
        
        has_data = False
        for idx, ticker_info in enumerate(tickers):
            ticker, name, unit = ticker_info
            df_combined = load_fred_data(ticker)
            if df_combined.empty:
                continue
            
            has_data = True

            if start_yr:
                df_combined = df_combined[df_combined.index >= f"{start_yr}-01-01"]
            if end_yr:
                df_combined = df_combined[df_combined.index <= f"{end_yr}-12-31"]

            color = colors_main[idx % len(colors_main)]
            
            # x축을 문자열로 변환하여 JSON 직렬화 지원 보장
            x_vals = df_combined.index.strftime('%Y-%m-%d').tolist()

            # 지표 트레이스 (왼쪽 y축)
            fig.add_trace(make_trace(chart_type, x_vals, df_combined[ticker].tolist(), name, color))

            # SEP Projections trace (If available)
            ticker_to_sep = {"GDPC1": "GDP", "UNRATE": "UNRATE", "PCEPI": "PCE", "FEDFUNDS": "FEDFUNDS"}
            sep_prefix = ticker_to_sep.get(ticker)
            if sep_prefix and SEP_LATEST_DATA:
                try:
                    sep_date = str(SEP_LATEST_DATA.get('Date', ''))
                    if sep_date and len(sep_date) >= 4:
                        base_year = int(sep_date[:4])
                        # SEP Year 1 is usually the meeting's current year end
                        x_proj = [f"{base_year}-12-31", f"{base_year+1}-12-31", f"{base_year+2}-12-31"]
                        y_proj = [
                            SEP_LATEST_DATA.get(f"{sep_prefix}_Year1"),
                            SEP_LATEST_DATA.get(f"{sep_prefix}_Year2"),
                            SEP_LATEST_DATA.get(f"{sep_prefix}_Year3")
                        ]
                        
                        x_clean, y_clean = [], []
                        for x_v, y_v in zip(x_proj, y_proj):
                            if pd.notna(y_v):
                                x_clean.append(x_v)
                                y_clean.append(float(y_v))
                                
                        if x_clean:
                            fig.add_trace(go.Scatter(
                                x=x_clean, y=y_clean,
                                name=f"{name} (SEP 전망)",
                                mode="lines+markers",
                                line=dict(color="#ef4444", width=2, dash='dot'),
                                marker=dict(symbol="circle", size=6, color="#ef4444")
                            ))
                except Exception as e:
                    print(f"Error adding SEP trace: {e}")

        if not has_data:
             return jsonify({"error": "No valid data to plot"}), 404

        layout = PLOTLY_LAYOUT.copy()
        
        title_name = ", ".join([t[1] for t in tickers])
        units = ", ".join(list(set([t[2] for t in tickers])))
        
        layout.update(
            title=dict(text=f"{title_name}", font=dict(color="#1f2937", size=12), x=0),
            yaxis=dict(title=dict(text=units, font=dict(size=12, color="#4b5563"))),
            height=320,
        )
        fig.update_layout(**layout)
        
        # 반환할 때 JSON 문자열로 캐스팅
        return fig.to_json()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
