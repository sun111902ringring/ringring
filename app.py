import datetime as dt
import json
import os
import time
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from folium.plugins import PolyLineTextPath   # 이동 흐름 화살표의 방향 표시(►)
from streamlit_folium import st_folium

# [역할] C안(날짜 예측) 모듈. [원리] 1년 전 기록 + 유사일 + 날씨 가정 + 달력으로 2026년 하루를 예측한다(longterm_c.py 참고).
import longterm_c as lc
# [역할] 기상청 단기·중기예보를 C안 날씨 입력으로 바꾸는 모듈(kma_weather.py 참고). 키가 없거나 실패하면 평년값으로 대신한다.
import kma_weather as kma


# ==========================================
# 0. 경로 및 상수
# ==========================================
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

SLICE_FILE = "app_slice.parquet"          # 앱용 소형 데이터 (feature 테이블의 일부)
STATIONS_FILE = "selected_stations_60.csv"  # 대여소 이름/구 정보

TARGET_RENTALS = "rentals_target_1h"
TARGET_RETURNS = "returns_target_1h"

MODEL_OPTIONS = ["GRU", "LightGBM", "XGBoost"]
PERIOD_OPTIONS = {"오전 출근": "morning", "오후 퇴근": "evening"}  # commute_period 값과 매칭
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
MAX_DAYS_AHEAD = 365  # 날짜 선택 가능 범위: 오늘 ~ 1년 뒤 (기존 '같은 요일 재생' 방식에서만 사용)
CLUSTER_VIEW_HEIGHT = 450  # 2번 메뉴: 지도와 표의 높이(px). 둘을 같은 값으로 맞춘다.

# [역할] 예측 방식 선택지와 C안 파일 위치.
# [원리] 기본은 C안(실제 날짜 예측). 기존 1시간 모델의 '같은 요일 재생'은 비교용으로 남긴다.
#        C안은 팀 목표(7일 이내 날짜 선택)에 맞춰 오늘부터 7일까지만 고를 수 있고, 데이터 끝(2025-12-31)에서 364일 뒤까지만 가능하다.
# [추가 2026-11-17] 1번 메뉴 이름(여러 곳에서 같은 글자를 써야 해서 상수로 둔다)
MENU1 = "1. 대여소 현황 지도 (예측)"
MENU5 = "5. 재배치 도우미 (따릉이 재배치 담당자용)"   # [추가 2026-11-17]
METHOD_LT = "날짜 예측 (작년 기록 + 날씨 · 추천)"
METHOD_REPLAY = "과거 같은 요일 재생 (1시간 모델 · 비교용)"
LT_MODEL_DIR = MODEL_DIR / "longterm_c"
LT_DATA_DIR = DATA_DIR / "longterm_c"
LT_DAYS_AHEAD = 7
LT_MODEL_LABEL = "C안 LightGBM"

# [추가 2026-11-17 이용자용 모델 이름] 날짜 예측 모델을 화면에 보여 줄 이름(실제 모델은 괄호에 병기).
# [원리] 이름에는 검증된 사실·역할만 넣는다(품질 약속 X). 기본값(LightGBM)은 '추천'으로 표시하고,
#        모델 간 차이가 작다는 점과 GRU의 약점은 이름이 아니라 도움말에 사실로 적는다. 내부 값(lt_model)은 그대로 영어 이름이다.
LT_MODEL_NAMES = {
    "LightGBM": "⭐ 추천 예측 (LightGBM)",
    "XGBoost": "비교 예측 (XGBoost)",
    "GRU": "시간 흐름 예측 (GRU)",
}
LT_MODEL_HELP = (
    "세 방식의 정확도 차이는 2% 이내라 기본값(추천 예측)을 쓰셔도 됩니다.\n\n"
    "- ⭐ 추천 예측: 시험에서 가장 정확했던 기본 방식이에요.\n"
    "- 비교 예측: 추천 예측과 거의 같은 정확도예요. 결과를 비교해 볼 때 쓰세요.\n"
    "- 시간 흐름 예측: 하루의 시간 흐름을 순서대로 학습한 방식이에요. 특이한 날씨에서는 결과가 크게 달라질 수 있어요."
)


def lt_name(model, star=True):
    """[역할] 날짜 예측 모델의 화면 이름. star=False면 제목용으로 ⭐를 뺀다."""
    name = LT_MODEL_NAMES.get(model, model)
    return name if star else name.replace("⭐ ", "")

# 자전거 경로: FOSSGIS가 운영하는 공개 OSRM 서버(자전거 프로필). 비상업·소규모 사용용이며 초당 1회 이하로 호출해야 한다.
ROUTING_BASE = "https://routing.openstreetmap.de/routed-bike"
ROUTING_USER_AGENT = "ddareungi-esg-demo/1.0"  # 서버 이용 규칙상 앱을 식별하는 값 (연락처를 덧붙이는 것을 권장)
ROUTING_TIMEOUT_S = 5
ROUTING_RETRY_AFTER_S = 300  # 경로 서버 호출이 실패하면 이 시간(초) 동안은 다시 부르지 않고 추정값을 쓴다
NEAREST_CANDIDATES = 3  # 출발지·도착지마다 보여 줄 가까운 대여소 후보 수

# 4번 메뉴(따릉이 길찾기): 주소 검색은 카카오 로컬 API
# 키 KAKAO_REST_KEY는 환경변수를 먼저 보고, 없으면 Streamlit secrets(.streamlit/secrets.toml 또는 Cloud Secrets)를 본다.
KAKAO_BASE = "https://dapi.kakao.com/v2/local/search"
KAKAO_TIMEOUT_S = 5
SEOUL_RECT = "126.76,37.41,127.19,37.72"  # 장소 이름 검색 범위: 서울을 감싸는 사각형 (경도,위도,경도,위도)
MAX_STATION_DIST_M = 2000   # 출발지·도착지에서 이 거리(직선)보다 먼 대여소는 후보에서 뺀다
WARN_STATION_DIST_M = 1000  # 이 거리보다 멀면 "대여소까지 멀다" 경고
BIKE_KMH_ESTIMATE = 15.0    # 경로 서버 실패 시 추정용 자전거 평균 속도 (가정값)
DETOUR_ESTIMATE = 1.3       # 경로 서버 실패 시 직선거리에 곱하는 우회 보정 (가정값)


# ==========================================
# 1. 페이지 및 기본 레이아웃 설정
# ==========================================
st.set_page_config(
    page_title="따릉이 출퇴근 수요 예측",
    page_icon="🚲",
    layout="wide",
)

with st.container(key="title_box"):
    st.title("🚲 AI 기반 따릉이 출퇴근 수요 예측 및 대여소 안내 서비스")
st.caption(
    # [수정 2026-11-17] 소개 문구를 이용자 말로
    "고른 날짜의 출퇴근 시간에 대여소마다 자전거가 남을지 모자랄지 예측하고, 대여소 유형 지도, "
    "시간대별 그래프, 가까운 대여소 길찾기를 제공합니다. "
    # [추가 2026-11-17] 대상이 60곳뿐임을 첫 화면에서 알린다(내 대여소가 목록에 없을 수 있음).
    "**분석 대상은 서울 따릉이 대여소 중 60곳**(역 주변·회사 지역·주거 지역 각 20곳)이에요."
)

# 사이드바: 접기(◀) 버튼을 '통합 운영 제어판' 제목과 같은 줄로 올리고, 내용을 위로 당긴다.
# (스트림릿 내부 구조(data-testid)에 의존하므로, 버전이 바뀌면 어긋날 수 있다)
st.markdown(
    """
    <style>
    .st-key-title_box {
        background: linear-gradient(-45deg, #fff1eb, #fce4ec, #e3f2fd, #e0f2f1);
        background-size: 400% 400%;
        animation: gradient 10s ease infinite;
        height: 100vh;
        padding: 15px 25px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
    }
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    [class*="st-key-subheader_box_"] {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 50%, #cffafe 100%);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
    }
    [data-testid="stHeading"] h1 {
        font-size: 2.25rem;
        text-align: center;
    }
    [data-testid="stHeading"] h3 {
        font-size: 1.5rem;
    }
    [data-testid="stSidebarHeader"] {
        position: absolute !important;
        top: calc(0.75rem + 20px) !important;
        right: 0.75rem !important;
        height: 2.25rem !important;
        min-height: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        z-index: 2;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: calc(0.75rem + 20px) !important;
        padding-bottom: 0.5rem !important;
    }
    .stat { 
        font-size: 1.2rem;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
    }
    .stat-row {
        display: grid;
        grid-template-columns: repeat(3, 7.5rem);
        column-gap: 0.5rem;
        justify-content: start;
        align-items: start;
    }
    .stat-label { font-size: 0.75rem; opacity: 0.75; white-space: nowrap; }
    .stat-help {
        display: inline-block;
        width: 0.95rem;
        height: 0.95rem;
        line-height: 0.95rem;
        text-align: center;
        font-size: 0.65rem;
        border: 1px solid currentColor;
        border-radius: 50%;
        opacity: 0.6;
        cursor: help;
        margin-left: 0.2rem;
    }
    .stat-line { display: flex; align-items: center; gap: 0.5rem; white-space: nowrap; }
    .stat-value { font-size: 1.2rem; }
    .stat-pill { font-size: 0.7rem; padding: 0.1rem 0.5rem; border-radius: 1rem; }
    .stat-pill.down { color: #09ab3b; background: rgba(33, 195, 84, 0.12); }
    .stat-pill.up { color: #ff2b2b; background: rgba(255, 43, 43, 0.12); }
    .compact-heading {
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.4;
        white-space: nowrap;
        margin: 0.25rem 0;
    }
    /* 사이드바 입력 상자: 번짐 없는 단색 그림자(0 3px 0 0)를 아래쪽에만 깔아 '아래 테두리'처럼 보이게 한다.
       드롭다운은 react-aria 구조라서 .react-aria-ComboBox 안의 role="group" 상자가 테두리 상자다.
       본문(메인 화면)의 드롭다운에는 적용되지 않도록 사이드바 안으로 범위를 좁혔다. */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"] {
        background-color: #fff;
    }
    /* 날짜 상자: 구조를 직접 확인하지 못해 추정한 선택자 (아래 세 가지 중 맞는 것이 적용된다) */
    [data-testid="stSidebar"] [data-testid="stDateInputField"] {
        background-color: #fff;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
        background-color: #fff;
    }
    [data-testid="stSidebarUserContent"] hr {
        margin: 1rem 0 !important;
    }
    [data-testid="stSidebarUserContent"] h3 {
        padding-top: 0.25rem !important;
    }
    [data-testid="stSidebarUserContent"] h2 {
        font-size: 1.25rem !important;
        line-height: 2.25rem !important;
        margin: 0 !important;
        padding: 0 3rem 0 0 !important;
        white-space: nowrap;
    }
    [data-testid="stSelectboxVirtualDropdown"] {
        background-color: #fff;
    }
    [data-testid="stTextInputRootElement"] {
        background-color: #fff;
    }
    .react-aria-ComboBox > [role="group"] {
        background-color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. 파일 위치 확인 및 데이터/모델 로딩 (캐싱 처리)
# ==========================================
def find_data_file(name):
    """data/ 폴더를 먼저, 없으면 app.py와 같은 폴더에서 찾는다."""
    for folder in (DATA_DIR, APP_DIR):
        path = folder / name
        if path.exists():
            return path
    return DATA_DIR / name


def model_path(model_name, target):
    if model_name == "GRU":
        return MODEL_DIR / f"bicycle_gru_{target}.keras"
    key = "lightgbm" if model_name == "LightGBM" else "xgboost"
    return MODEL_DIR / f"bicycle_{key}_{target}.pkl"


def missing_files(model_name):
    meta_name = "gru_meta.pkl" if model_name == "GRU" else "feature_meta.pkl"
    needed = [
        find_data_file(SLICE_FILE),
        find_data_file(STATIONS_FILE),
        model_path(model_name, TARGET_RENTALS),
        model_path(model_name, TARGET_RETURNS),
        MODEL_DIR / meta_name,
    ]
    return [p for p in needed if not p.exists()]


@st.cache_data(show_spinner=False)
def load_slice():
    df = pd.read_parquet(find_data_file(SLICE_FILE))
    df["station_id"] = df["station_id"].astype(str)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df


def read_stations_csv():
    path = find_data_file(STATIONS_FILE)
    try:
        stations = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        stations = pd.read_csv(path, encoding="cp949")
    stations["station_id"] = stations["station_id"].astype(str)
    # [추가 2026-11-17 최신 이름] 화면 이름만 2026년 공식 이름으로 바꿔 끼운다(원래 목록 파일·좌표·예측은 그대로).
    # [원리] scripts/build_station_names_2026.py가 만든 data/station_names_2026.csv의 display_name을 대여소 번호로 붙인다.
    #        파일이 없거나 이름이 비어 있으면 원래 이름을 그대로 쓴다.
    names_path = DATA_DIR / "station_names_2026.csv"
    if names_path.exists():
        latest = pd.read_csv(names_path, encoding="utf-8-sig", dtype={"station_id": str}).set_index("station_id")["display_name"]
        stations["station_name"] = stations["station_id"].map(latest).fillna(stations["station_name"])
    return stations


@st.cache_data(show_spinner=False)
def load_stations():
    return read_stations_csv()[["station_id", "station_name", "gu", "group"]]


# [추가 2026-11-17 대여소 유형] 메뉴 2에서 쓰는 대여소 유형표와 설명.
# [역할] scripts/build_station_types.py가 미리 계산해 둔 K-Means 결과(대여소별 유형, 유형별 이름·색·설명)를 읽는다.
# [원리] 유형은 2025년 평일 기록으로 한 번만 계산해 파일로 둔다. 앱은 읽기만 하므로 빠르고, 볼 때마다 결과가 같다.
#        파일이 없으면 None을 돌려 화면에서 만드는 방법을 안내한다.
STATION_TYPES_FILE = DATA_DIR / "station_types.csv"
STATION_TYPES_META = DATA_DIR / "station_types_meta.json"


# [추가 2026-11-17 시간대별 이용 가능성] 내 대여소 카드에 '작년 기록 기준 빌리기·반납하기 가능성'을 시간대별로 보여 준다.
# [역할] 실시간 앱에 없는 '몇 시에 가면 수월한가'를 알려 준다(출발 시각·대여소 고르기에 도움).
# [원리] scripts/build_station_use_freq.py가 만든 작년(2025) 빈도표를 읽는다.
#        고른 날짜 d → 작년 기준일(d − 364일)·날 종류(평일 / 주말·공휴일)·시각으로 찾아,
#        빈도 < 0.3 🟢, < 0.6 🟠, 그 이상 🔴. 비교한 날이 5일 미만이면 판정하지 않는다.
#        검증(2023→2024, 2024→2025): 대여 🟢일 때 실제로 빈 비율 11%, 🔴 62~63% / 반납(거치대 1.5배 이상) 🟢 15%, 🔴 70~73%.
#        그날 날씨·예측은 섞지 않는다(섞으면 검증에서 오히려 덜 맞았다).
USE_FREQ_FILE = DATA_DIR / "station_use_freq.parquet"
USE_RENT_TEXT = ["🟢 수월해요", "🟠 빠듯할 수 있어요", "🔴 자전거가 없을 수 있어요"]
USE_RETURN_TEXT = ["🟢 심하게 붐비진 않아요", "🟠 붐빌 수 있어요", "🔴 매우 붐빌 수 있어요"]


@st.cache_data(show_spinner=False)
def load_use_freq():
    if not USE_FREQ_FILE.exists():
        return None
    f = pd.read_parquet(USE_FREQ_FILE)
    f["station_id"] = f["station_id"].astype(str)
    return f.set_index(["station_id", "hour", "kind", "ref_date"])


@st.cache_data(show_spinner=False)
def load_holiday_set():
    path = LT_DATA_DIR / "holidays.csv"
    return set(pd.read_csv(path)["date"]) if path.exists() else set()


def use_level(freq):
    return 2 if freq >= 0.6 else 1 if freq >= 0.3 else 0


# [추가 2026-11-17 대여소 검색] 표를 대여소 이름·옛 이름·번호로 찾아 좁혀 보는 기능.
# [역할] 60곳 표에서 내 대여소를 바로 찾게 한다. 개명 전 이름(예: '청구역')이나 따릉이 앱에 보이는 번호로도 찾을 수 있다.
# [원리] 대여소마다 '검색용 글자'(지금 이름 + 2021년 이름 + 2026년 이름 + 번호)를 띄어쓰기 없이 소문자로 만들어 두고,
#        검색어도 같은 방식으로 바꿔 '포함되는지'만 본다. 그래서 '3번 출구'와 '3번출구'가 같게 찾아진다.
def _norm(text):
    return "".join(str(text).split()).lower()


@st.cache_data(show_spinner=False)
def load_station_search_text():
    """결과: station_id → 검색용 글자. 이름표 파일(station_names_2026.csv)이 없으면 지금 이름과 번호만 쓴다."""
    st_df = read_stations_csv()[["station_id", "station_name"]]
    words = {sid: [name, sid] for sid, name in zip(st_df["station_id"], st_df["station_name"])}
    names_path = DATA_DIR / "station_names_2026.csv"
    if names_path.exists():
        nm = pd.read_csv(names_path, encoding="utf-8-sig", dtype={"station_id": str})
        for _, row in nm.iterrows():
            words.setdefault(row["station_id"], []).extend([row.get("name_2021", ""), row.get("name_2026", "")])
    return {sid: _norm(" ".join(str(w) for w in ws if pd.notna(w))) for sid, ws in words.items()}


def station_search(df, key, id_col="대여소ID"):
    """[역할] 검색칸을 그리고, 검색어와 맞는 대여소만 남긴 표를 돌려준다. 검색어가 없으면 그대로 돌려준다."""
    q = st.text_input("🔎 대여소 검색", key=key, placeholder="이름·옛 이름·번호 (예: 천호역, 청구역, 1012)",
                      help="띄어쓰기는 무시해요. 이름이 바뀐 대여소는 옛 이름으로도 찾을 수 있어요.")
    if not q.strip():
        return df
    text = load_station_search_text()
    nq = _norm(q)
    found = df[df[id_col].astype(str).map(lambda sid: nq in text.get(sid, ""))]
    if found.empty:
        st.caption(f"'{q}'와(과) 맞는 대여소가 없어요. 분석 대상은 서울 따릉이 대여소 중 60곳이라, 찾는 대여소가 없을 수도 있어요.")
    else:
        st.caption(f"'{q}' 검색 결과 {len(found)}곳")
    return found


def times_text(freq, what):
    """[역할] 빈도를 쉬운 말로. [원리] 10번 중 0번·10번은 어색해 '거의 없었어요'·'거의 매번'으로 바꾼다."""
    n = round(freq * 10)
    if n == 0:
        return f"작년엔 {what}인 적이 거의 없었어요"
    if n == 10:
        # 받침 있는 글자(예: '이상') 뒤에는 '이었어요', 없으면(예: '이하') '였어요'
        ending = "이었어요" if (ord(what[-1]) - 0xAC00) % 28 else "였어요"
        return f"작년엔 거의 매번 {what}{ending}"
    return f"작년 10번 중 {n}번은 {what}"


def show_use_forecast(station_id, date, hours):
    """[역할] 고른 대여소의 시간대별 빌리기·반납하기 가능성 표를 그린다. 빈도표가 없으면 아무것도 그리지 않는다.
    [수정 2026-11-18] 🔴(어려움)인 시각 목록을 돌려준다: {"rent": [빌리기 🔴 시각], "return": [반납하기 🔴 시각]}.
    [왜] 카드 맨 아래 안내가 위쪽 예측 상태만 보고 '큰 문제 없음'이라고 하는데, 이 표에는 🔴가 있는 경우가 있었다(다음 7일 예측의 약 7%).
         안내 문구가 이 표의 🔴도 함께 보고 정하도록 결과를 넘겨준다. 빈도표가 없으면 None."""
    freq = load_use_freq()
    if freq is None:
        return None
    kind = "off" if (date.weekday() >= 5 or date.strftime("%Y-%m-%d") in load_holiday_set()) else "work"
    ref = pd.Timestamp(date) - pd.Timedelta(days=364)
    rows = ["| 시간대 | 🚲 빌리기 | 🅿️ 반납하기 |", "|---|---|---|"]
    red = {"rent": [], "return": []}
    for h in hours:
        key = (str(station_id), h, kind, ref)
        if key not in freq.index or freq.loc[key, "n_ref"] < 5:
            rows.append(f"| {h}시대 | 기록이 적어 판단하기 어려워요 | - |")
            continue
        fe, fc = float(freq.loc[key, "f_empty"]), float(freq.loc[key, "f_crowd"])
        if use_level(fe) == 2:
            red["rent"].append(h)
        if use_level(fc) == 2:
            red["return"].append(h)
        rows.append(f"| {h}시대 ({h}:00~{h}:59) | {USE_RENT_TEXT[use_level(fe)]}<br>"
                    f"<small>{times_text(fe, '3대 이하')}</small> | "
                    f"{USE_RETURN_TEXT[use_level(fc)]}<br><small>{times_text(fc, '거치대의 1.5배 이상')}</small> |")
    kind_text = "평일" if kind == "work" else "주말·공휴일"
    st.markdown(f"##### ⏱️ 시간대별 이용 가능성 (작년 이 시기 {kind_text} 실제 기록 기준)")
    st.markdown("\n".join(rows), unsafe_allow_html=True)
    st.caption(
        "작년 같은 시기(앞뒤 4주) 같은 날 종류의 실제 재고로 판정했어요. 지난 두 해로 시험해 보니, 빌리기 🟢일 때 실제로 빈 경우는 "
        "약 10번 중 1번, 🔴일 때는 10번 중 6번쯤이었어요. 반납하기의 '매우 붐빔'은 자전거가 거치대의 1.5배 이상 쌓인 상태로, "
        "그래도 다른 자전거에 연결해 반납할 수 있어요(🟢도 '자리 있음'이 아니라 '심하게 붐비진 않음'이에요). "
        "날씨나 그날 상황은 반영하지 않아 평일끼리는 비슷한 결과가 나와요. 지도의 '포화(거치대 거의 참)'와는 기준이 달라요.")
    names_path = DATA_DIR / "station_names_2026.csv"
    if names_path.exists():
        note = pd.read_csv(names_path, encoding="utf-8-sig", dtype={"station_id": str}).set_index("station_id")["note"]
        if "위치 이전 여부 미확인" in str(note.get(str(station_id), "")):
            st.caption("⚠️ 이 대여소는 2026년에 이름이 바뀌어, 작년 기록과 지금 위치가 다를 수 있어요.")
    return red


@st.cache_data(show_spinner=False)
def load_station_types():
    if not (STATION_TYPES_FILE.exists() and STATION_TYPES_META.exists()):
        return None, None
    types = pd.read_csv(STATION_TYPES_FILE, encoding="utf-8-sig", dtype={"station_id": str})
    meta = json.loads(STATION_TYPES_META.read_text(encoding="utf-8"))
    return types, meta


# [추가 2026-11-17 표기 통일] 화면의 날짜·강수 표기를 한 가지로 맞춘다.
# [원리] 날짜는 '11/17(화)', 강수는 '비 없음 / 약한 비 / 보통 비 / 강한 비'(입력칸에서만 mm 범위까지 자세히).
def day_text(d):
    return f"{d:%m/%d}({WEEKDAYS[d.weekday()]})"


def rain_short(i):
    return "비 없음" if int(i) == 0 else lc.PRECIP_LABELS[int(i)].split(" (")[0]


def inv_text(inv, dock):
    """[역할] 예측 끝 재고를 화면용 글로. [원리] 예측 재고는 상한 없이 계산해 거치대를 크게 넘을 수 있지만(예: 108대),
    실제로는 그 전에 재배치가 일어나 숫자 자체는 과장일 수 있다. 그래서 거치대 수를 넘으면 숫자 대신
    '거치대 초과'로만 알리고, 넘지 않으면 '약 N대'로 보여 준다(판정에는 원래 숫자를 그대로 쓴다)."""
    # [수정 2026-11-18 숫자 대신 구간 (B)]
    # [왜] 2024·2025년 검증에서 예측 끝 재고는 실제와 평균 약 10대 차이가 났다(실제 평균 17~19대). '약 7대'처럼 쓰면
    #      정확한 값처럼 보이므로, 숫자 대신 판정 기준과 같은 경계로 나눈 구간 이름을 보여 준다.
    # [원리] 3대 이하 '거의 없음' / 거치대 절반 미만 '적은 편' / 거치대 − 3 미만 '넉넉한 편' / 거치대까지 '거치대 거의 참' / 넘으면 '거치대 초과'.
    if pd.isna(inv) or pd.isna(dock):
        return "-"
    if inv > dock:
        return "거치대 초과"
    if inv <= 3:
        return "거의 없음"
    if inv < dock / 2:
        return "적은 편"
    if inv < dock - 3:
        return "넉넉한 편"
    return "거치대 거의 참"


def flow_text(ratio):
    """[역할] '반납 ÷ 대여' 배수를 누구나 아는 말로. [원리] 1보다 크면 들어오는 쪽, 작으면 나가는 쪽으로 읽어
    '1대 나갈 때 N대 들어와요'처럼 큰 쪽을 N으로 말한다(0.28배보다 '3.6대 나가요'가 이해하기 쉽다)."""
    if ratio >= 1.2:
        return f"자전거 1대가 나갈 때 **{ratio:.1f}대가 들어와요**"
    if ratio <= 1 / 1.2:
        return f"자전거 1대가 들어올 때 **{1 / ratio:.1f}대가 나가요**"
    return "들어오는 양과 나가는 양이 **비슷해요**"


@st.cache_data(show_spinner=False)
def load_station_locations():
    """5번 메뉴용: 대여소 이름·구·좌표 (모델/예측 데이터 없이 CSV만 사용)."""
    stations = read_stations_csv()[["station_id", "station_name", "gu", "lat", "lon"]]
    return stations.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def pick_reference_date(target_date, df):
    """선택한 날짜와 같은 요일인 '가장 최근' 실제 데이터 날짜를 고른다.
    평일은 휴일이 아닌 날을 우선한다. (공휴일 달력이 2025년까지라 선택일의 공휴일 여부는 반영하지 않음)"""
    dates = pd.DataFrame({"date": sorted(df["date"].unique())})
    dates["weekday"] = [d.weekday() for d in dates["date"]]
    if "is_offday" in df.columns:
        offday = df.groupby("date")["is_offday"].max().fillna(False).astype(bool)
        dates["offday"] = dates["date"].map(offday).astype(bool)
    else:
        dates["offday"] = False

    candidates = dates[dates["weekday"] == target_date.weekday()]
    if target_date.weekday() < 5:
        non_off = candidates[~candidates["offday"]]
        if not non_off.empty:
            candidates = non_off
    return None if candidates.empty else candidates["date"].max()


def sidebar_station_picker(disabled=False, where=None):
    """사이드바 대여소 드롭다운. 선택한 station_id를 돌려주고, '전체 대여소'면 None.
    where: 위젯을 놓을 칸(기본은 사이드바). [수정 2026-11-17] 사이드바 정리용."""
    where = st.sidebar if where is None else where
    if not find_data_file(STATIONS_FILE).exists():
        return None
    stations = load_stations()
    names = stations["station_name"].fillna(stations["station_id"])
    duplicated = names.duplicated(keep=False)  # 이름이 같은 대여소는 ID를 덧붙여 구분
    labels = np.where(duplicated, names + " (" + stations["station_id"] + ")", names)
    lookup = dict(zip(labels, stations["station_id"]))
    choice = where.selectbox(
        "📍 대여소 선택",
        ["전체 대여소"] + sorted(lookup),
        disabled=disabled,
        help="1번 지도는 고른 대여소로 확대되고, 2번 메뉴에는 '내 대여소 카드'가, 3번 그래프에는 그 대여소만 나오며, "
             "5번 이동 흐름의 기준 대여소가 됩니다. "
             "4번 길찾기에서는 쓰지 않습니다.",
    )
    return None if choice == "전체 대여소" else lookup[choice]


@st.cache_resource(show_spinner="모델을 불러오는 중입니다...")
def load_model(model_name, target):
    path = model_path(model_name, target)
    if model_name == "GRU":
        from tensorflow import keras  # GRU 선택 시에만 불러와 시작 속도 확보

        return keras.models.load_model(path, compile=False)
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_meta(model_name):
    meta_name = "gru_meta.pkl" if model_name == "GRU" else "feature_meta.pkl"
    return joblib.load(MODEL_DIR / meta_name)


# ==========================================
# 3. 모델 입력 만들기 + 예측
#    (학습 스크립트와 동일한 전처리를 그대로 재현)
# ==========================================
def build_tree_features(df, meta):
    """LightGBM / XGBoost: 학습 때와 같은 category 목록으로 변환 (NaN은 그대로 둠)."""
    X = df[meta["inputs"]].copy()
    for col, cats in meta["categories"].items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


def build_gru_inputs(df, meta):
    """GRU: bool→float, 숫자 결측 0 대체, 범주형은 학습 때 저장한 정수 코드로 변환."""
    num_cols, cat_cols, cat_maps = meta["num_cols"], meta["cat_cols"], meta["cat_maps"]
    X = df[num_cols + cat_cols].copy()

    for col in num_cols:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype("float64")
    for col in num_cols:
        X[col] = X[col].fillna(0.0).astype("float64")

    inputs = {}
    for col in cat_cols:
        s = X[col].astype(str).fillna("missing")
        unknown_code = len(cat_maps[col])  # Embedding(input_dim=num_classes + 1)의 여분 칸
        codes = s.map(cat_maps[col]).fillna(unknown_code).astype("int64")
        inputs[f"input_{col}"] = codes.values
    inputs["input_numeric"] = X[num_cols].values
    return inputs


def predict_rows(model_name, target, df):
    model = load_model(model_name, target)
    meta = load_meta(model_name)
    if model_name == "GRU":
        pred = model.predict(build_gru_inputs(df, meta), batch_size=4096, verbose=0)
        pred = np.asarray(pred).reshape(-1)
    else:
        pred = model.predict(build_tree_features(df, meta))
    return np.clip(np.asarray(pred, dtype=float), 0, None)  # 학습 평가와 동일하게 음수는 0으로


@st.cache_data(show_spinner="선택한 모델로 예측 중입니다...")
def run_prediction(model_name, date, period_label):
    """선택한 날짜·피크 시간대의 대여소별 대여/반납을 예측하고 Net-flow를 계산한다."""
    period = PERIOD_OPTIONS[period_label]
    df = load_slice()
    sub = df[(df["date"] == date) & (df["commute_period"].astype(str) == period)].copy()
    if sub.empty:
        return None, {}

    sub = sub.sort_values(["station_id", "timestamp"])
    sub["pred_rentals"] = predict_rows(model_name, TARGET_RENTALS, sub)
    sub["pred_returns"] = predict_rows(model_name, TARGET_RETURNS, sub)
    return summarize_period(sub, model_name)


# [수정 2026-11-17 포화 기준 확정] 포화 판정은 옛 기준(예측 재고 ≥ 거치대 수 - 3)을 유지하고 '뜻'만 바로잡는다.
# [왜] 2025년 검증(experiments/longterm_c/eval_full_threshold.py)에서 옛 기준이 실제와 가장 잘 맞았다(F1 출근 0.77·퇴근 0.74).
#      대여소별 상위 10% 같은 상대 기준은 표시는 적지만 실제 포화의 80~90%를 놓쳤다.
#      거치대가 차도 다른 자전거에 연결해 반납하는 일이 흔하므로(2025년 시간의 45%가 재고 > 거치대),
#      '반납 불가'가 아니라 '거치대 거의 참'으로 표시한다.
# [추가] 상시 과밀: 2025년 평소(중앙값) 재고가 거치대 수보다 많은 대여소. '매일 그런 곳'과 '오늘 특히 그런 곳'을 구분하려고 쓴다.
@st.cache_data(show_spinner=False)
def load_chronic_full():
    """결과: station_id → 2025년 재고 중앙값(대). 파일이 없으면 빈 Series(상시 과밀 표시를 생략)."""
    path = LT_DATA_DIR / "history_hourly.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    h = pd.read_parquet(path, columns=["station_id", "timestamp", "inventory"])
    h = h[h["timestamp"].dt.year == 2025].dropna(subset=["inventory"])
    return h.groupby(h["station_id"].astype(str))["inventory"].median()


# [추가 2026-11-18 부족 판정 보강 (A)]
# [역할] '자전거 부족 예상'이 실제 부족을 많이 놓치던 문제를 줄인다.
# [왜] 2024·2025년 검증(experiments/longterm_c/eval_status_accuracy.py)에서 예측 재고만 보는 판정은 실제 부족의
#      32~42%만 잡았다(시작 재고로 쓰는 작년 평균이 부정확해서). 재고 추정을 쓰지 않는 '작년 빈도'와 둘 중 하나라도
#      위험하면 부족으로 보면 잡는 비율이 55~63%로 오르고, 맞는 비율은 52~56%로 비슷했다(F1 0.41~0.49 → 0.53~0.59, 두 해 같은 결과).
# [원리] 작년 빈도 = 작년 같은 시기(고른 날 − 364일 ± 28일)·같은 날 종류(평일 / 주말·공휴일)에, 시간대가 끝난 직후 시각
#        (출근 10시, 퇴근 21시)의 재고가 3대 이하였던 날의 비율. 비교한 날이 5일 미만인 대여소는 빈도를 쓰지 않는다.
#        빈도 ≥ 0.5(작년 이 시기 절반 이상의 날)면 '작년 기록상 부족'. 0.5는 검증 전에 정한 값이다.
@st.cache_data(show_spinner=False)
def _end_hour_inventory():
    """결과: 출퇴근이 끝난 직후 시각(10·21시)의 작년 재고표(station_id, day, hour, inventory). 파일이 없으면 빈 표."""
    path = LT_DATA_DIR / "history_hourly.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["station_id", "day", "hour", "inventory"])
    h = pd.read_parquet(path, columns=["station_id", "timestamp", "inventory"]).dropna(subset=["inventory"])
    ts = h["timestamp"].dt.tz_localize(None)
    h = h.assign(station_id=h["station_id"].astype(str), day=ts.dt.normalize(), hour=ts.dt.hour)
    return h[h["hour"].isin([10, 21])][["station_id", "day", "hour", "inventory"]]


@st.cache_data(show_spinner=False)
def last_year_short_freq(date, end_h):
    """결과: station_id → 작년 이 시기 끝 시각 재고가 3대 이하였던 날의 비율(비교일 5일 이상인 대여소만)."""
    h = _end_hour_inventory()
    h = h[h["hour"] == end_h]
    ref = pd.Timestamp(date) - pd.Timedelta(days=364)
    h = h[(h["day"] - ref).abs() <= pd.Timedelta(days=28)]
    hol = load_holiday_set()
    off = (h["day"].dt.day_of_week >= 5) | h["day"].dt.strftime("%Y-%m-%d").isin(hol)
    target_off = date.weekday() >= 5 or date.strftime("%Y-%m-%d") in hol
    g = (h.loc[off == target_off, "inventory"] <= 3).groupby(h.loc[off == target_off, "station_id"])
    n, f = g.size(), g.mean()
    return f[n >= 5]


# [역할] 시간별 예측(sub)을 대여소별 표(지도·표·길찾기에서 쓰는 형태)로 요약한다.
# [원리] 피크 시간대 3시간의 예측 대여·반납을 더하고, 시간대 시작 재고 + (반납 - 대여)로 끝날 때 재고를 추정해
#        3대 이하면 부족, 거치대 수 - 3 이상이면 포화(거치대 거의 참)로 표시한다. 기존 1시간 모델과 C안(날짜 예측)이 함께 쓰도록 run_prediction에서 분리했다.
#        sub에 필요한 열: station_id, timestamp, hour, latitude, longitude, dock_count, inventory_lag_1h, pred_rentals, pred_returns
#        (실제값 열·기상 열은 있으면 쓰고 없으면 건너뛴다)
# [수정 2026-11-18] short_freq(작년 빈도, last_year_short_freq 결과)를 받으면 부족 판정에 함께 쓴다(날짜 예측일 때만 넘긴다).
#        같은 요일 재생(비교용)은 참조일이 과거라 '작년 빈도'를 만들 자료가 없어 예전 규칙 그대로 둔다.
def summarize_period(sub, model_name, short_freq=None):
    sub = sub.sort_values(["station_id", "timestamp"])
    # 대여소별로 피크 시간대 전체(시간별 예측의 합)를 집계
    spec = {
        "위도": ("latitude", "first"),
        "경도": ("longitude", "first"),
        "총거치대수": ("dock_count", "first"),
        "현재자전거수": ("inventory_lag_1h", lambda s: s.iloc[0]),  # 시간대 시작 시점의 재고
        "예측_대여": ("pred_rentals", "sum"),
        "예측_반납": ("pred_returns", "sum"),
    }
    has_actual = {TARGET_RENTALS, TARGET_RETURNS}.issubset(sub.columns)
    if has_actual:
        spec["실제_대여"] = (TARGET_RENTALS, lambda s: s.sum(min_count=1))
        spec["실제_반납"] = (TARGET_RETURNS, lambda s: s.sum(min_count=1))
    out = sub.groupby("station_id").agg(**spec).reset_index()

    excluded = int(out[["현재자전거수", "총거치대수"]].isna().any(axis=1).sum())
    out = out.dropna(subset=["현재자전거수", "총거치대수"]).copy()

    out["순수요_NetFlow"] = out["예측_반납"] - out["예측_대여"]  # 반납 - 대여
    if has_actual:
        out["실제_NetFlow"] = out["실제_반납"] - out["실제_대여"]
    else:
        out["실제_NetFlow"] = np.nan

    # [수정 2026-11-17] 예측 재고를 거치대 수에서 자르지 않는다(거치대보다 많이 놓이는 것이 실제로 흔함). 0 아래만 막는다.
    out["예측_잔여수량"] = (
        (out["현재자전거수"] + out["순수요_NetFlow"])
        .round()
        .clip(lower=0)
        .astype(int)
    )
    # 상시 과밀 여부(평소 재고 > 거치대). 기록이 없는 대여소는 False.
    usual = out["station_id"].astype(str).map(load_chronic_full())
    out["평소재고"] = usual.round()
    out["상시과밀"] = (usual > out["총거치대수"]).fillna(False).astype(bool)

    # [수정 2026-11-18 부족 판정 보강] 부족 = 예측 끝 재고 ≤ 3대 '또는' 작년 빈도 ≥ 0.5 (위 last_year_short_freq 설명 참고).
    #        어느 근거로 부족이 됐는지(부족근거)를 남겨 카드·지도 설명에 쓴다.
    out["부족_작년빈도"] = (out["station_id"].astype(str).map(short_freq) if short_freq is not None else np.nan)
    pred_short = out["예측_잔여수량"] <= 3
    freq_short = (out["부족_작년빈도"] >= 0.5).fillna(False).astype(bool)
    out["부족근거"] = np.select([pred_short & freq_short, pred_short, freq_short],
                              ["예측·작년 기록", "예측", "작년 기록"], default="")
    conditions = [
        pred_short | freq_short,
        (out["예측_잔여수량"] >= out["총거치대수"] - 3),
    ]
    # [수정 2026-11-17] '반납 불가'는 과장 → 거치대가 거의 차서 다른 자전거에 연결해 반납해야 할 수 있다는 뜻으로.
    # [수정 2026-11-18] '대여 불가'는 예측을 단정하는 말이라 '자전거 거의 없음'으로 바꿈(판정 범위를 넓힌 뒤 더 어색해져서).
    #        화면 곳곳은 앞부분('부족'·'포화')으로 상태를 구분하므로 괄호 안 문구만 바꿔도 동작은 같다.
    choices = ["부족 위험 🔴 (자전거 거의 없음)", "포화 위험 🔵 (거치대 거의 참)"]
    out["위험상태"] = np.select(conditions, choices, default="정상 🟢")
    out["적용모델"] = model_name

    # 대여소 이름/구 정보 붙이기
    out = out.merge(load_stations(), on="station_id", how="left")
    unmatched = int(out["station_name"].isna().sum())
    out["대여소ID"] = out["station_id"]
    out["대여소명"] = out["station_name"].fillna(out["station_id"])
    out["클러스터"] = out["gu"].fillna("미상")  # 클러스터 = 자치구
    out["구분"] = out["group"].fillna("미상")
    out = out.drop(columns=["station_name", "gu", "group"])

    # 관측 기상 (모델 입력으로 쓰인 1시간 전 관측값; 서울 단일 관측소라 모든 대여소가 같음). C안처럼 열이 없으면 결측.
    has_wx = {"temp_c_lag1h", "precip_mm_lag1h"}.issubset(sub.columns)
    wx = sub.groupby("timestamp").agg(
        temp=("temp_c_lag1h", "mean"), precip=("precip_mm_lag1h", "mean")
    ) if has_wx else pd.DataFrame({"temp": [np.nan], "precip": [np.nan]})
    info = {
        "hour_min": int(sub["hour"].min()),
        "hour_max": int(sub["hour"].max()),
        "temp": float(wx["temp"].mean()),
        "precip": float(wx["precip"].sum(min_count=1)),
        "excluded": excluded,
        "unmatched": unmatched,
        "has_actual": has_actual,
    }
    return out, info


@st.cache_data(show_spinner="선택한 모델로 하루치 예측 중입니다...")
def run_daily_prediction(model_name, date):
    """참조일 하루(0~23시) 전체의 대여소별 시간당 대여/반납을 예측한다."""
    df = load_slice()
    sub = df[df["date"] == date].copy()
    if sub.empty:
        return None

    sub["pred_rentals"] = predict_rows(model_name, TARGET_RENTALS, sub)
    sub["pred_returns"] = predict_rows(model_name, TARGET_RETURNS, sub)

    sub = sub.merge(load_stations(), on="station_id", how="left")
    sub["대여소ID"] = sub["station_id"]
    sub["대여소명"] = sub["station_name"].fillna(sub["station_id"])
    return sub[["대여소ID", "대여소명", "hour", "pred_rentals", "pred_returns"]]


# ==========================================
# 3-1. C안(날짜 예측) 연결
# ==========================================
# [역할] C안에 필요한 파일이 모두 있는지 확인. [원리] 없으면 화면에 어떤 파일이 빠졌는지 보여 주고 그 화면만 멈춘다.
LT_FILES = [LT_MODEL_DIR / n for n in ("model_rentals.txt", "model_returns.txt", "preprocess.json", "meta.json")] + \
           [LT_DATA_DIR / n for n in ("history_hourly.parquet", "daily_weather.csv", "station_static.csv", "holidays.csv")]


def lt_files_missing():
    return [p for p in LT_FILES if not p.exists()]


# [역할] C안 모델·자료를 한 번만 읽어 둔다. [원리] st.cache_resource는 앱이 켜져 있는 동안 같은 객체를 재사용한다.
@st.cache_resource(show_spinner="날짜 예측 모델을 불러오는 중입니다...")
def load_lt_bundle():
    return lc.load_bundle(LT_MODEL_DIR, LT_DATA_DIR)


# [역할] 고른 날짜·날씨 가정으로 60곳 × 24시간 예측. [원리] 같은 날짜·같은 날씨면 다시 계산하지 않도록 결과를 캐시한다.
#        weather_items는 (이름, 값) 튜플들 — 캐시 키로 쓰려면 바뀌지 않는(해시 가능한) 형태여야 한다.
# [추가 2026-11-16] 날짜 예측용 XGBoost·GRU 모델 읽기. [원리] 고른 모델만 처음 한 번 읽어 재사용한다(GRU는 TensorFlow라 느림).
@st.cache_resource(show_spinner="선택한 예측 모델을 준비하는 중이에요. 처음 한 번만 10초 정도 걸릴 수 있어요...")
def load_lt_extra(model_name):
    return lc.load_extra(LT_MODEL_DIR, model_name)


# [역할] 고른 날짜·날씨·모델로 60곳 × 24시간 예측. [원리] 같은 날짜·같은 날씨·같은 모델이면 다시 계산하지 않도록 캐시한다.
#        weather_items는 (이름, 값) 튜플들 — 캐시 키로 쓰려면 바뀌지 않는(해시 가능한) 형태여야 한다.
@st.cache_data(show_spinner="날짜 예측 모델로 하루치를 계산하는 중입니다...")
def run_lt_day(date, weather_items, model_name="LightGBM"):
    extra = None if model_name == "LightGBM" else load_lt_extra(model_name)
    day = lc.predict_day(date, dict(weather_items), load_lt_bundle(), model_name, extra)
    return day, list(day.attrs.get("similar_dates", []))


# [역할] C안 결과를 기존 지도·표·길찾기와 같은 형태(대여소별 표 + info)로 만든다.
# [원리] 피크 시간대 3시간만 골라 summarize_period로 요약. 시작 재고는 실제 재고가 없으므로
#        '작년 같은 시기 같은 시각 평균 재고'(inventory_ly)로 대신한다 → 화면에 추정값이라고 표시한다.
def run_prediction_lt(date, period_label, weather_items, model_name="LightGBM"):
    day, sims = run_lt_day(date, weather_items, model_name)
    static = load_lt_bundle().static[["station_id", "latitude", "longitude", "dock_count"]]
    sub = day[day["commute_period"] == PERIOD_OPTIONS[period_label]].merge(static, on="station_id", how="left")
    sub = sub.rename(columns={"inventory_ly": "inventory_lag_1h"})
    # [수정 2026-11-18] 시간대가 끝난 직후 시각(출근 10시·퇴근 21시)의 작년 부족 빈도를 함께 넘긴다(부족 판정 보강).
    short_freq = last_year_short_freq(date, int(sub["hour"].max()) + 1)
    out, info = summarize_period(sub, lt_name(model_name, star=False), short_freq)   # [수정 2026-11-17] 화면용 이름
    info["similar_dates"] = sims
    return out, info


# [역할] 3번 메뉴(시간대별 그래프)용 하루치 표. [원리] 기존 run_daily_prediction과 같은 열 이름으로 돌려준다.
def run_daily_lt(date, weather_items, model_name="LightGBM"):
    day, sims = run_lt_day(date, weather_items, model_name)
    day = day.merge(load_stations(), on="station_id", how="left")
    day["대여소ID"] = day["station_id"]
    day["대여소명"] = day["station_name"].fillna(day["station_id"])
    return day[["대여소ID", "대여소명", "hour", "pred_rentals", "pred_returns"]], sims


# [역할] C안 파일이 없으면 안내 후 멈춘다(예측이 필요한 화면 전용).
def require_lt():
    missing = lt_files_missing()
    if missing:
        st.error("날짜 예측에 필요한 파일이 없습니다. 아래 파일을 넣거나 사이드바에서 '과거 같은 요일 재생'을 고르세요.")
        st.code("\n".join(str(p.relative_to(APP_DIR)) for p in missing))
        st.stop()
    # [추가 2026-11-17] 예측 가능 기한이 지났으면 계산하지 않고 안내만 한다(그대로 계산하면 기록이 없어 오류가 난다).
    if lt_expired:
        st.error(f"📅 날짜 예측은 {lt_last:%Y-%m-%d}까지만 할 수 있어요. 예측에 쓰는 작년 기록(2025년)의 범위를 넘었기 때문입니다. "
                 "2026년 이용 기록으로 예측 자료를 갱신하면 다시 쓸 수 있어요. 지금은 사이드바에서 "
                 "'과거 같은 요일 재생(비교용)'을 고르면 참고용 화면을 볼 수 있습니다.")
        st.stop()


# [역할] 기상청 예보 받기(30분마다 새로). [원리] time_bucket이 30분 단위로 바뀌면 캐시가 새로 계산된다.
#        성공한 결과만 캐시한다: 실패하면 예외를 던지고(Streamlit은 예외를 캐시하지 않음) 바깥에서 받아 평년값으로 대신한다.
#        그래서 일시적인 실패나 키 등록 직후에도 30분을 기다리지 않고 다음 화면 갱신 때 다시 시도한다.
@st.cache_data(show_spinner="기상청 예보를 받는 중입니다...", ttl=1800)
def _kma_forecast_cached(time_bucket):
    table, notes = kma.fetch_forecast(kma.service_key())   # 키는 함수 안에서 읽어 캐시 키에 남기지 않는다
    if not table:
        raise RuntimeError("; ".join(notes) or "예보 없음")
    return table, notes


# [추가 2026-11-18 기상청 실패 잠시 기억]
# [역할] 기상청 서버가 응답 없이 멈출 때, 화면을 누를 때마다 다시 기다리느라 앱이 느려지는 것을 막는다.
# [왜] 실패는 캐시하지 않으므로(위 설명) 매 클릭마다 최대 4번 × 8초(약 30초)를 기다릴 수 있었다.
# [원리] 마지막 실패 시각과 이유를 앱 전체가 함께 쓰는 상자(st.cache_resource)에 적어 두고,
#        KMA_RETRY_AFTER_S(5분) 안에는 기상청을 다시 부르지 않고 바로 평년값으로 넘어간다. 5분이 지나면 다시 시도한다.
#        성공하면 상자를 비운다. 키가 없을 때는 원래대로 바로 안내한다(기다림 없음).
KMA_RETRY_AFTER_S = 300


@st.cache_resource(show_spinner=False)
def _kma_fail_box():
    return {"at": 0.0, "reason": ""}


def load_kma_forecast(time_bucket):
    """결과: (날짜별 예보 표, 실패 메모). 키가 없거나 실패하면 빈 표와 이유를 돌려준다."""
    if not kma.service_key():
        return {}, ["기상청 서비스 키(KMA_SERVICE_KEY)가 없습니다."]
    box = _kma_fail_box()
    wait = KMA_RETRY_AFTER_S - (time.time() - box["at"])
    if wait > 0:
        # 화면 안내는 앞 120자만 보이므로 '언제 다시 시도하는지'를 앞에 둔다.
        return {}, [f"약 {max(1, round(wait / 60))}분 뒤 다시 시도해요", box["reason"]]
    try:
        result = _kma_forecast_cached(time_bucket)
        box["at"], box["reason"] = 0.0, ""
        return result
    except Exception as e:  # noqa: BLE001 — 이유를 화면 설명에 보여 준다
        box["at"], box["reason"] = time.time(), str(e)
        return {}, [str(e)]


# [수정 2026-11-17 이용자용 안내] 날짜 예측 결과 위의 안내 상자.
# [역할] 이용자에게는 '무엇을 보고 예측했고, 얼마나 믿으면 되는지'만 쉬운 말로 보여 주고,
#        팀 내부 용어·계산식(모델 이름, 비슷한 날 목록, ×보정값, 시험 점수 원값)은 '예측 방법 자세히 보기'(기본 닫힘)로 옮긴다.
# [원리] 시험 오차·경향은 고른 모델의 2025년 테스트 결과(meta.json test_2025_by_model)를 반올림해 쓴다.
#        Bias가 +면 '조금 많게', -면 '조금 적게' 예측하는 경향으로 표현한다.
def show_lt_banner(sims):
    wx = dict(weather_items)
    sim_text = ", ".join(str(s) for s in sims) if sims else "없음(후보 부족)"
    factor = lc.h2_factor(selected_date, load_lt_bundle(), "rentals")
    tst = load_lt_bundle().meta.get("test_2025_by_model", {}).get(lt_model)
    src = str(wx.get("source", ""))
    src_easy = ("직접 고친 날씨" if "사용자 수정" in src else "기상청 예보" if src.startswith("기상청")
                else "평년 날씨(예보를 받지 못함)")

    rain = rain_short   # [수정 2026-11-17] 공용 함수로 통일

    rain_words = "눈" if wx.get("snow") and (wx["precip_am_cat"] or wx["precip_pm_cat"]) else None
    weather_easy = (f"최저 {wx['temp_min']:.0f}℃ / 최고 {wx['temp_max']:.0f}℃ · "
                    + (f"{rain_words} 예보" if rain_words else f"오전 {rain(wx['precip_am_cat'])} · 오후 {rain(wx['precip_pm_cat'])}"))
    # 소수 둘째 자리까지: 첫째 자리로 반올림하면 1.444(1.4)와 1.451(1.5)처럼 작은 차이가 크게 보인다.
    mae = f"{tst['rentals_MAE']:.2f}" if tst else "1.4"
    lean = "조금 많게" if (tst is None or tst["rentals_Bias"] >= 0) else "조금 적게"
    st.info(
        f"📅 **{selected_date:%m/%d}({WEEKDAYS[selected_date.weekday()]}) 예측** — 작년 같은 시기의 이용 기록과 "
        f"이날 날씨({src_easy} · {weather_easy})를 바탕으로 예측했어요.\n\n"
        f"**참고하실 점**\n"
        f"- 실제와 시간당 평균 **약 {mae}대** 차이가 날 수 있어요(지난해 기록으로 시험한 결과).\n"
        f"- 실제보다 **{lean}** 예측하는 경향이 있어요.\n"
        # [수정 2026-11-18] 재고 추정 오차(평균 약 10대)와 구간 표시·부족 판정 보강을 알린다.
        f"- 남은 자전거 수는 작년 같은 시각의 평균으로 추정해 실제와 평균 약 10대 차이가 나요. 그래서 숫자 대신 "
        f"'거의 없음·적은 편·넉넉한 편'처럼 구간으로 보여 드려요.\n"
        f"- '자전거 부족 예상'은 예측과 작년 이 시기 기록 중 하나라도 부족하면 표시해요(실제 부족을 더 많이 미리 알리기 위해서)."
    )
    # [추가 2026-11-17] 오늘 날짜를 고른 이용자는 '지금 재고'로 오해하기 쉬워 더 눈에 띄게 알린다.
    # [원리] 실시간 재고 API를 연결하지 않았으므로, 남은 자전거 수는 오늘도 작년 평균 기반 추정값이다.
    if selected_date == today:
        st.warning("⚠️ **오늘 날짜라도 실시간 재고가 아니에요.** 남은 자전거 수는 작년 같은 시각 평균으로 추정한 값입니다. "
                   "지금 실제로 몇 대 있는지는 따릉이 앱에서 확인해 주세요.")
    with st.expander("🔎 예측 방법 자세히 보기"):
        lines = [
            f"- **예측 방식:** {lt_name(lt_model, star=False)}. 1년 전(52주 전) 기록, 작년 비슷한 날씨의 날, 날씨, 달력(요일·공휴일), "
            "대여소 정보를 입력으로 학습한 모델입니다(서비스용 모델은 2022~2025년 기록으로 학습).",
            f"- **작년 비슷한 날씨의 날:** {sim_text}",
            f"- **날씨 입력:** {src or '가정'} — 최저 {wx['temp_min']}℃ / 최고 {wx['temp_max']}℃, "
            f"오전 {lc.PRECIP_LABELS[wx['precip_am_cat']]}, 오후 {lc.PRECIP_LABELS[wx['precip_pm_cat']]}"
            f"{', 강수 형태 눈' if wx.get('snow') else ''}{', 전날 눈' if wx.get('snow_prev') else ''}",
        ]
        if factor != 1:
            lines.append(f"- **연간 이용 변화 보정:** 2026년 상반기 서울 따릉이 이용이 전년보다 줄어든 비율의 절반을 반영해 "
                         f"예측에 ×{factor:.3f}를 곱했습니다.")
        if tst:
            lines.append(f"- **2025년 시험 결과(한 번만 평가):** 시간당 평균 오차 대여 {tst['rentals_MAE']:.3f}대 · 반납 "
                         f"{tst['returns_MAE']:.3f}대, 평균 치우침 대여 {tst['rentals_Bias']:+.3f}대(+는 많게 예측).")
        lines.append("- **남은 자전거 수:** 실시간 재고가 없으므로 작년 같은 시기 같은 시각의 평균 재고에 예측 (반납 − 대여)를 더해 추정합니다.")
        # [추가 2026-11-18] 부족 판정 보강의 근거(검증 숫자)를 자세히 보기에 남긴다.
        lines.append("- **자전거 부족 판정:** 예측 끝 재고 3대 이하 **또는** 작년 같은 시기(±4주)·같은 날 종류에 시간대가 끝날 때 "
                     "3대 이하였던 날이 절반 이상. 2024·2025년 기록으로 시험했을 때 예측만 쓰면 실제 부족의 32~42%를 미리 잡았고, "
                     "둘을 함께 쓰면 55\~63%를 잡았어요(표시가 맞은 비율은 52\~56%로 비슷). 남은 자전거 추정 오차는 평균 약 10대예요.")
        st.markdown("\n".join(lines))
    # [추가 2026-11-16 눈] 눈 예보일(전날 눈 포함) 불확실성 안내.
    # [원리] 학습 기간 5년(1,826일) 중 눈 추정일은 83일뿐이고, 모델은 적설량(1cm인지 10cm인지)을 모른다.
    #        그래서 눈 오는 날 예측은 평소보다 크게 빗나갈 수 있음을 숫자와 함께 알린다.
    if lc.is_snow_day(wx):
        st.warning(
            "❄️ **눈 예보일(또는 전날 눈)입니다.** 모델이 배운 눈 오는 날은 5년 중 83일뿐이고, 적설량(1cm인지 10cm인지)을 "
            "구분하지 못합니다. 빙판·폭설이면 실제 이용은 이 예측보다 훨씬 적을 수 있으니 참고용으로만 보세요."
            + (" 최저기온이 영상인 눈은 학습 자료에 없던 조합이라 더 불확실합니다." if wx.get("snow") and wx["temp_min"] > 0 else "")
        )


def haversine_m(lat1, lon1, lat2, lon2):
    """두 지점 사이의 직선거리(m). lat2/lon2는 배열이어도 된다."""
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    d_phi = p2 - p1
    d_lambda = np.radians(lon2) - np.radians(lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(d_lambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_to_point(stations, lat, lon, k=NEAREST_CANDIDATES, max_m=MAX_STATION_DIST_M):
    """임의 위치에서 직선거리로 가까운 대여소 k곳 (max_m 이내만). 없으면 빈 표."""
    out = stations.copy()
    out["straight_m"] = haversine_m(lat, lon, out["lat"].to_numpy(dtype=float), out["lon"].to_numpy(dtype=float))
    return out[out["straight_m"] <= max_m].nsmallest(k, "straight_m").reset_index(drop=True)


def one_based(df):
    """표의 행 번호(맨 왼쪽 열)가 0이 아니라 1부터 시작하도록 바꾼다."""
    out = df.reset_index(drop=True)
    out.index = out.index + 1
    return out


def fmt_distance(meters):
    return f"{meters / 1000:.1f} km" if meters >= 1000 else f"{meters:,.0f} m"


def osrm_get(path, params):
    resp = requests.get(
        f"{ROUTING_BASE}/{path}",
        params=params,
        headers={"User-Agent": ROUTING_USER_AGENT},
        timeout=ROUTING_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise ValueError(f"routing error: {data.get('code')} {data.get('message', '')}")
    return data


@st.cache_data(show_spinner="자전거 경로를 찾는 중입니다...", ttl=24 * 3600)
def bike_route(a_lat, a_lon, b_lat, b_lon):
    """대여소 → 대여소 자전거 경로. 실패하면 예외를 던진다(실패 결과가 캐시되지 않도록)."""
    try:
        data = osrm_get(
            f"route/v1/driving/{a_lon:.6f},{a_lat:.6f};{b_lon:.6f},{b_lat:.6f}",
            {"overview": "full", "geometries": "geojson"},
        )
    finally:
        time.sleep(1.1)  # 공개 서버 이용 규칙: 초당 1회 이하
    route = data["routes"][0]
    return {
        "distance_m": float(route["distance"]),
        "duration_s": float(route["duration"]),
        "estimated": False,
        "path": [[la, lo] for lo, la in route["geometry"]["coordinates"]],  # GeoJSON은 (경도, 위도) 순서
    }


def bike_leg(a, b):
    """경로 서버를 먼저 쓰고, 실패하면 직선거리×보정 ÷ 평균속도로 추정한다.
    실패한 뒤 ROUTING_RETRY_AFTER_S 동안은 서버를 다시 부르지 않아 화면이 느려지지 않게 한다."""
    if time.time() >= st.session_state.get("routing_down_until", 0):
        try:
            return bike_route(a[0], a[1], b[0], b[1])
        except Exception:
            st.session_state["routing_down_until"] = time.time() + ROUTING_RETRY_AFTER_S
    dist = float(haversine_m(a[0], a[1], b[0], b[1])) * DETOUR_ESTIMATE
    return {"distance_m": dist, "duration_s": dist / (BIKE_KMH_ESTIMATE * 1000 / 3600),
            "estimated": True, "path": [list(a), list(b)]}


def fit_view(points, width_px=600, height_px=480, padding_px=50, max_zoom=17):
    """점들이 모두 보이는 지도 중심과 확대 단계(웹 메르카토르 기준).
    width_px는 지도 실제 폭을 알 수 없어 좁은 화면 기준으로 잡은 값이다(넓으면 여유가 더 생길 뿐)."""
    lats = np.array([p[0] for p in points], dtype=float)
    lons = np.array([p[1] for p in points], dtype=float)
    merc = np.log(np.tan(np.pi / 4 + np.radians(lats) / 2))  # 위도 → 메르카토르 y
    dx = (lons.max() - lons.min()) / 360.0
    dy = (merc.max() - merc.min()) / (2 * np.pi)
    zooms = [max_zoom]
    if dx > 0:
        zooms.append(np.log2((width_px - 2 * padding_px) / (256 * dx)))
    if dy > 0:
        zooms.append(np.log2((height_px - 2 * padding_px) / (256 * dy)))
    zoom = int(np.clip(np.floor(min(zooms)), 3, max_zoom))
    center_lat = float(np.degrees(2 * np.arctan(np.exp((merc.max() + merc.min()) / 2)) - np.pi / 2))
    return [center_lat, float((lons.max() + lons.min()) / 2)], zoom


def kakao_key():
    """환경변수 → Streamlit secrets 순서로 카카오 REST API 키를 찾는다. 없으면 None."""
    key = os.environ.get("KAKAO_REST_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["KAKAO_REST_KEY"]).strip() or None
    except Exception:  # secrets.toml이 없거나 키가 없음
        return None


@st.cache_data(show_spinner="주소를 찾는 중입니다...", ttl=24 * 3600)
def kakao_search(query):
    """장소 이름 → 주소 순으로 검색. 결과 [{name, address, lat, lon}] (최대 5개, 서울 주소 우선).
    장소 이름은 서울 범위(SEOUL_RECT) 안에서만 찾는다(예: '중앙시장'이 광주로 잡히지 않도록).
    오류는 requests.HTTPError 등 예외로 던진다(캐시되지 않도록)."""
    headers = {"Authorization": f"KakaoAK {kakao_key()}"}
    for kind, extra in (("keyword", {"rect": SEOUL_RECT}), ("address", {})):
        resp = requests.get(f"{KAKAO_BASE}/{kind}.json", headers=headers,
                            params={"query": query, "size": 5, **extra}, timeout=KAKAO_TIMEOUT_S)
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        if docs:
            docs = sorted(docs, key=lambda d: not d.get("address_name", "").startswith("서울"))  # 서울 먼저(순서 유지)
            return [{
                "name": d.get("place_name") or d.get("address_name", query),
                "address": d.get("road_address_name") or d.get("address_name", ""),
                "lat": float(d["y"]), "lon": float(d["x"]),  # 카카오는 x=경도, y=위도
            } for d in docs]
    return []


def station_forecast():
    """사이드바 조건(모델·날짜·피크 시간대)의 대여소별 예측 상태. (표 또는 None, 안내 문구).
    예측이 안 돼도 길찾기는 계속되도록 st.stop() 없이 None을 돌려준다."""
    # [역할] 4번 메뉴(길찾기)의 재고 상태. [원리] C안이면 날짜 예측 결과를 쓰고, 시작 재고가 추정값임을 안내에 적는다.
    if use_lt:
        if lt_files_missing():
            return None, "날짜 예측 파일이 없어 재고 예측 상태는 표시하지 않습니다."
        try:
            df_pred, _ = run_prediction_lt(selected_date, period_label, weather_items, lt_model)
        except Exception:
            return None, "예측 중 오류가 나서 재고 예측 상태는 표시하지 않습니다."
        note = (f"재고 상태는 {day_text(selected_date)} {period_label} 날짜 예측(날씨 예보 반영) 기준이며, 시작 재고는 "
                "작년 같은 시각 평균으로 추정한 참고값입니다. 실제 실시간 재고가 아닙니다.")
        return df_pred.set_index("대여소ID")[["위험상태", "예측_잔여수량", "총거치대수"]], note
    if missing_files(model_type):
        return None, "예측 모델·데이터 파일이 없어 재고 예측 상태는 표시하지 않습니다."
    try:
        ref_date = pick_reference_date(selected_date, load_slice())
        if ref_date is None:
            return None, "참조할 같은 요일 데이터가 없어 재고 예측 상태는 표시하지 않습니다."
        df_pred, _ = run_prediction(model_type, ref_date, period_label)
    except Exception:
        return None, "예측 중 오류가 나서 재고 예측 상태는 표시하지 않습니다."
    if df_pred is None:
        return None, "참조일에 해당 시간대 데이터가 없어 재고 예측 상태는 표시하지 않습니다."
    note = (f"재고 상태는 {day_text(selected_date)} {period_label} 예측(같은 요일 참조일 {ref_date} 데이터, {model_type}) 기준의 "
            "참고용 시뮬레이션입니다. 실제 실시간 재고가 아닙니다.")
    return df_pred.set_index("대여소ID")[["위험상태", "예측_잔여수량", "총거치대수"]], note


def hourly_totals(daily, station_id=None):
    """시간(0~23)별 예측 대여/반납 합계. station_id가 없으면 전체 대여소 합계."""
    sub = daily if station_id is None else daily[daily["대여소ID"] == station_id]
    hourly = sub.groupby("hour")[["pred_rentals", "pred_returns"]].sum().reindex(range(24))
    hourly.columns = ["예측 대여", "예측 반납"]
    hourly.index.name = "시간"
    return hourly


# ==========================================
# 4. 사이드바 제어판
# ==========================================
# [수정 2026-11-17 사이드바 정리] 처음 온 이용자가 위에서부터 '메뉴 → 날짜·시간대·대여소'만 고르면 되게 한다.
# [원리] Streamlit은 미리 만든 칸(container)에 나중에 위젯을 채울 수 있다. 보이는 순서대로 칸을 먼저 만들고
#        (기본 조건 → 날씨 → 고급 설정), 계산 순서상 먼저 필요한 '예측 방식·모델'은 맨 아래 접힌 '고급 설정' 칸에 채운다.
#        같은 요일 재생(비교용)·모델 고르기는 일반 이용자가 몰라도 되므로 고급 설정 안에 둔다.
st.sidebar.header("🚲 무엇을 볼까요?")
mode = st.sidebar.radio(
    "📋 메뉴 선택",
    [
        # [수정 2026-11-17] 예측 화면임이 이름에 드러나게(사용자 결정)
        MENU1,   # [수정 2026-11-17] 실시간 아님(예측) + 전문 용어 제거
        # [수정 2026-11-17] 메뉴 이름의 기술 꼬리표((K-Means)·(선 그래프)·(경로 지도)) 제거. K-Means는 2번 화면의 '분석 방법'에 남긴다.
        "2. 대여소 유형 지도 (출퇴근 흐름별)",
        "3. 시간대별 대여·반납 예측 그래프",
        "4. 가까운 대여소 길찾기",
        MENU5,   # [추가 2026-11-17] 재배치 담당자용 화면 분리
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 언제, 어디를 볼까요?")
sb_basic = st.sidebar.container()     # 날짜·출퇴근 시간대·대여소
sb_weather = st.sidebar.container()   # 날씨 자동 반영·날씨 고치기·날짜별 예보
sb_adv = st.sidebar.expander("⚙️ 고급 설정 (예측 방식·모델)", expanded=False)
# [역할] 예측 방식 선택. [원리] C안은 고른 날짜를 실제로 예측하고, 기존 방식은 같은 요일 과거 하루를 1시간 모델로 재생한다.
method = sb_adv.radio(
    "📈 예측 방식", [METHOD_LT, METHOD_REPLAY],
    # [수정 2026-11-17] 도움말을 이용자 말로
    help="날짜 예측(추천): 작년 같은 시기 기록과 그날 날씨로 고른 날짜를 예측합니다.\n\n"
         "같은 요일 재생(비교용): 고른 날짜와 같은 요일의 지난 실제 하루를 다시 계산해 보는 시뮬레이션입니다.",
)
use_lt = method == METHOD_LT
lt_missing = lt_files_missing() if use_lt else []
# [수정 2026-11-16] 예측 모델 선택. [원리] 날짜 예측이면 날짜 예측용 모델(LightGBM·XGBoost·GRU 중 파일이 있는 것),
#        같은 요일 재생이면 기존 1시간 모델(GRU·LightGBM·XGBoost) 목록을 보여 준다. 두 목록은 서로 다른 모델이다.
lt_model = "LightGBM"
if use_lt:
    lt_models = ["LightGBM"] if lt_missing else lc.available_models(LT_MODEL_DIR)
    # [수정 2026-11-17] 선택지는 내부 이름(LightGBM 등) 그대로 두고, 화면에는 이용자용 이름으로 보여 준다(format_func).
    lt_model = sb_adv.selectbox("🤖 예측 모델 선택", lt_models, key="lt_model_pick",
                                    format_func=lt_name, help=LT_MODEL_HELP)
    model_type = lt_name(lt_model, star=False)   # 화면 제목·지도 key에 표시되는 이름 (기존 모델 경로에서는 쓰이지 않음)
else:
    model_type = sb_adv.selectbox("🤖 예측 모델 선택", MODEL_OPTIONS, index=MODEL_OPTIONS.index("LightGBM"),
                                      help="같은 요일 재생에 쓰는 1시간 예측 모델입니다.")
# [추가 2026-11-17] 화면 제목용 이름: 날짜 예측은 '추천 예측 (LightGBM)'처럼 그대로, 같은 요일 재생은 'LightGBM 모델'.
model_title = model_type if use_lt else f"{model_type} 모델"

today = pd.Timestamp.now(tz="Asia/Seoul").date()
# [테스트용] 환경변수 RINGRING_TODAY=YYYY-MM-DD 가 있으면 그 날을 '오늘'로 본다(기한 안내 화면 점검용, 평소엔 없음).
if os.environ.get("RINGRING_TODAY"):
    today = dt.date.fromisoformat(os.environ["RINGRING_TODAY"])
# [역할] 날짜 범위. [원리] C안은 오늘~7일(팀 목표) 그리고 모델이 계산 가능한 마지막 날(2026-12-30) 중 이른 날까지.
# [수정 2026-11-17 기한 안내] 날짜 예측은 2025년 기록 + 364일까지만 가능하다.
#   ① 기한 14일 전부터: 고를 수 있는 날짜가 줄어든다는 것을 미리 알린다.
#   ② 기한이 지나면: 날짜 예측 화면(메뉴 1~3)을 멈추고 이유와 대안(같은 요일 재생)을 안내한다(lt_expired → require_lt).
lt_expired = False
if use_lt:
    lt_last = dt.date(2026, 12, 30) if lt_missing else dt.date.fromisoformat(load_lt_bundle().meta["max_target_date"])
    lt_max = min(today + dt.timedelta(days=LT_DAYS_AHEAD), lt_last)
    if lt_max < today:
        lt_expired = True
        sb_basic.error(f"날짜 예측은 {lt_last:%Y-%m-%d}까지만 할 수 있어요. 새 이용 기록으로 예측 자료를 갱신해야 합니다.")
        lt_max = today
    elif (lt_last - today).days < 14:
        sb_basic.warning(f"⏳ 날짜 예측은 {lt_last:%m/%d}까지만 할 수 있어요(2025년 기록 기준). "
                           f"그 뒤 날짜는 고를 수 없습니다.")
    selected_date = sb_basic.date_input(f"☀️ 날짜 선택 (오늘부터 {LT_DAYS_AHEAD}일 이내)", value=today,
                                          min_value=today, max_value=lt_max)
else:
    selected_date = sb_basic.date_input(
        "☀️ 날짜 선택 (오늘 이후)",
        value=today,
        min_value=today,
        max_value=today + dt.timedelta(days=MAX_DAYS_AHEAD),
    )
period_label = sb_basic.selectbox("⏰ 출퇴근 시간대 선택", list(PERIOD_OPTIONS))
selected_station_id = sidebar_station_picker(disabled=mode.startswith("4."), where=sb_basic)

# [역할] 날씨 입력(C안 전용). [원리] ① 기상청 예보가 있으면 그 값을 기본값으로 채운다(단기예보 우선, 없으면 중기예보).
#        ② 키가 없거나 그날 예보가 없으면 2021~2025년 같은 시기(±7일) 평균 기온 + '비 없음'(평년값)을 쓴다.
#        ③ 사용자가 값을 고치면 그 값으로 예측하고 출처를 '수정됨'으로 표시한다.
#        위젯 key에 날짜·출처를 넣어, 날짜나 예보가 바뀌면 기본값이 새로 채워지게 한다.
weather_items = None
if use_lt and not lt_missing:
    now_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).to_pydatetime()
    fc_table, fc_notes = load_kma_forecast(now_kst.strftime("%Y%m%d%H") + ("30" if now_kst.minute >= 30 else "00"))
    fc = kma.to_model_weather(selected_date, fc_table)
    if fc:
        d_tmin, d_tmax, d_am, d_pm, d_snow, src = (fc["temp_min"], fc["temp_max"], fc["precip_am_cat"],
                                                    fc["precip_pm_cat"], fc["snow_prev"], f"기상청 {fc['source']}")
        d_is_snow = int(fc.get("snow", 0))
    else:
        (d_tmin, d_tmax), d_am, d_pm, d_snow, src = lc.climatology(load_lt_bundle().daily, selected_date), 0, 0, 0.0, "평년값"
        d_is_snow = 0
    tag = f"{selected_date}_{src}"
    title = "🌦️ 날씨 (기상청 예보)" if fc else "🌦️ 날씨 가정 (평년값)"
    wx_info = sb_weather.container()   # [수정 2026-11-17] 자동 반영 상자를 '날씨 고치기' 메뉴보다 위에 보이게 하는 칸
    with sb_weather.expander(title.replace("🌦️ 날씨", "✏️ 날씨 고치기"), expanded=False):
        # [수정 2026-11-16] 기온은 소수점 첫째 자리까지만 입력·표시(기상청 예보·학습 자료도 0.1℃ 단위). +/- 버튼은 0.1℃씩.
        t_min = st.number_input("🧊 최저기온 (℃)", value=round(float(d_tmin), 1), step=0.1, format="%.1f", key=f"lt_tmin_{tag}")
        t_max = st.number_input("🔥 최고기온 (℃)", value=round(float(d_tmax), 1), step=0.1, format="%.1f", key=f"lt_tmax_{tag}")
        am = st.selectbox("🌅 오전 강수", range(4), index=int(d_am), format_func=lambda i: lc.PRECIP_LABELS[i], key=f"lt_am_{tag}")
        pm = st.selectbox("🌇 오후 강수", range(4), index=int(d_pm), format_func=lambda i: lc.PRECIP_LABELS[i], key=f"lt_pm_{tag}")
        # [추가 2026-11-16 눈] 강수 형태. [원리] '눈'을 고르면 기온과 관계없이 눈으로 예측한다(강수량이 '없음'이면 약한 눈으로 처리).
        #        '비'면 학습 때 규칙대로 최저기온 0℃ 이하일 때만 눈으로 추정한다. 기상청이 눈을 예보하면 '눈'이 기본값.
        snow = st.selectbox("❄️ 강수 형태", [0, 1], index=d_is_snow, key=f"lt_snow_{tag}",
                            format_func=lambda i: "눈" if i else "비 (최저기온 0℃ 이하면 눈으로 자동 판단)")
        if fc:
            st.caption(f"출처: {src}. 값을 고치면 고친 값으로 예측합니다. 모델은 실제 날씨로 학습되어, 예보가 틀리면 예측도 틀릴 수 있습니다.")
        else:
            st.caption("기상청 예보를 쓰지 못해 2021~2025년 같은 시기 평균 기온(평년값)과 '비 없음'을 기본값으로 넣었습니다."
                       + (f" ({'; '.join(fc_notes)[:160]})" if fc_notes else ""))
    # 기온은 0.1℃ 단위로 반올림해서 비교·사용한다(+/- 버튼을 누르면 8.200000001 같은 오차가 생길 수 있어서).
    t_min, t_max = round(float(t_min), 1), round(float(t_max), 1)
    edited = (t_min, t_max, int(am), int(pm), int(snow)) != \
        (round(float(d_tmin), 1), round(float(d_tmax), 1), int(d_am), int(d_pm), int(d_is_snow))
    weather_items = (("temp_min", float(t_min)), ("temp_max", float(t_max)),
                     ("precip_am_cat", int(am)), ("precip_pm_cat", int(pm)), ("snow_prev", float(d_snow)),
                     ("snow", int(snow)), ("source", src + (" · 사용자 수정" if edited else "")))
    # [추가 2026-11-17 예보 자동 반영 표시]
    # [역할] 날씨 칸(접힌 메뉴)을 열지 않아도 '어떤 날씨가 자동으로 들어갔는지'를 사이드바에 항상 보여 준다.
    # [원리] src에는 예보 종류와 발표 시각이 들어 있다(예: '기상청 단기예보 11/17 05시 발표').
    #        중기예보 날짜는 강수량 없이 강수확률(60% 이상 = 약한 비)로만 넣으므로 정확도가 낮다는 안내를 덧붙이고,
    #        예보를 못 받아 평년값을 쓰면 그 사실과 이유를 경고로 보여 준다.
    if fc:
        wx_info.success(
            f"🛰️ **날씨 자동 반영** · {src}\n\n최저 {t_min}℃ / 최고 {t_max}℃ · 오전 {rain_short(am)} · "
            f"오후 {rain_short(pm)}{' · 눈' if snow else ''}"
            + ("\n\n✏️ 사용자가 값을 수정해 수정한 값으로 예측합니다." if edited else "")
        )
        if "중기예보" in src:
            wx_info.warning("이 날짜는 중기예보라 강수량 대신 **강수확률**(60% 이상이면 '약한 비')로 넣었습니다. "
                               "강한 비가 와도 '약한 비'로 들어가므로 단기예보 날짜보다 정확도가 낮습니다.")
    else:
        wx_info.warning("⚠️ 기상청 예보를 받지 못해 **평년값**(같은 시기 평균 기온, 비 없음)으로 예측합니다."
                           + (f" ({'; '.join(fc_notes)[:120]})" if fc_notes else ""))
    # [역할] 선택 가능한 모든 날짜에 어떤 예보가 자동으로 들어가는지 한눈에 보여 준다(발표 자료용으로도 사용).
    # [원리] 이미 받아 둔 예보 표(fc_table)를 날짜마다 모델 입력 형태(to_model_weather)로 바꿔 표로 만든다.
    #        새 API 호출은 없다. 예보가 없는 날은 '평년값'으로 표시한다(실제 예측도 그 날은 평년값을 쓴다).
    if fc_table:
        rows = []
        for i in range((lt_max - today).days + 1):
            d = today + dt.timedelta(days=i)
            w = kma.to_model_weather(d, fc_table)
            rows.append({
                "날짜": f"{d:%m/%d}({WEEKDAYS[d.weekday()]})",
                "최저·최고(℃)": f"{w['temp_min']:.1f} / {w['temp_max']:.1f}" if w else "-",
                "오전": rain_short(w["precip_am_cat"]) if w else "-",
                "오후": rain_short(w["precip_pm_cat"]) if w else "-",
                "눈": "눈" if w and w.get("snow") else "",
                "출처": (w["source"].split(" ")[0] if w else "평년값"),   # '단기예보' / '중기예보' / '평년값'
            })
        with sb_weather.expander("📅 날짜별 자동 반영 예보", expanded=False):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("단기예보: 시간별 강수량을 오전·오후로 합산 · 중기예보: 강수확률·날씨 문구로 비 여부만 판단. "
                       "모델은 실제 날씨로 학습되어 예보가 틀리면 예측도 함께 틀립니다.")
    sb_adv.caption(
        # [수정 2026-11-17] 팀 내부 용어(C안)·시험 용어를 빼고 쉬운 말로
        "날짜 예측: 작년 같은 시기의 이용 기록과 날씨로 예측합니다. "
        "지난해 기록으로 시험했을 때, 작년 평균을 그대로 쓰는 것보다 약 21% 더 정확했어요."
    )
elif use_lt:
    sb_basic.error("날짜 예측 파일이 없습니다. models/longterm_c, data/longterm_c 폴더를 확인하세요.")
else:
    # [추가 2026-11-17] 고급 설정이 접혀 있어도 비교 모드임을 알 수 있게 위쪽에 표시
    sb_basic.info("🔁 비교용 '같은 요일 재생' 모드입니다. 고급 설정에서 날짜 예측으로 바꿀 수 있어요.")
    sb_adv.caption(
        "선택한 날짜는 아직 실제 데이터가 없으므로, 같은 요일의 가장 최근 실제 데이터(대여·반납 이력, 재고, 기상)를 "
        "모델에 입력해 예측하는 시뮬레이션입니다."
    )


def get_reference_date():
    """파일 확인 → 참조일 선택 → 시뮬레이션 안내 배너. 예측이 필요한 모드가 공통으로 사용."""
    problems = missing_files(model_type)
    if problems:
        st.error("예측에 필요한 파일을 찾을 수 없습니다. 아래 위치에 파일이 있는지 확인해 주세요.")
        st.code("\n".join(str(p.relative_to(APP_DIR)) if APP_DIR in p.parents else str(p) for p in problems))
        st.stop()

    ref_date = pick_reference_date(selected_date, load_slice())
    if ref_date is None:
        st.warning(
            "선택한 날짜와 같은 요일의 참조 데이터가 없습니다. "
            "app_slice.parquet에 최소 2주 이상의 데이터가 들어 있는지 확인해 주세요."
        )
        st.stop()

    st.info(
        f"🧪 **시뮬레이션 결과입니다.** 선택한 {selected_date}({WEEKDAYS[selected_date.weekday()]})은 "
        f"아직 실제 데이터가 없어, 같은 요일인 **{ref_date}**의 실제 데이터(대여·반납 이력, 재고, 기상)를 "
        f"{model_type} 모델에 입력해 예측했습니다. 실제 기상 예보가 아니며, 공휴일은 반영되지 않습니다."
    )
    return ref_date


def get_predictions():
    # [역할] 1·2번 메뉴 공통 예측. [원리] C안이면 날짜 예측, 아니면 기존 '같은 요일 재생'을 쓴다. 돌려주는 형태는 같다.
    if use_lt:
        require_lt()
        df_pred, info = run_prediction_lt(selected_date, period_label, weather_items, lt_model)
        show_lt_banner(info["similar_dates"])
        if info["excluded"]:
            st.caption(f"⚠️ 작년 같은 시각 재고 기록이 없는 대여소 {info['excluded']}곳은 지도·표에서 제외했습니다.")
        return df_pred, info
    ref_date = get_reference_date()
    df_pred, info = run_prediction(model_type, ref_date, period_label)
    if df_pred is None:
        st.warning(f"참조일({ref_date})에 해당 피크 시간대 데이터가 없습니다. 다른 시간대를 선택해 주세요.")
        st.stop()
    info["ref_date"] = ref_date

    if info["excluded"]:
        st.caption(f"⚠️ 재고 정보가 없는 대여소 {info['excluded']}곳은 제외했습니다.")
    if info["unmatched"]:
        st.caption(f"⚠️ 대여소 이름 파일과 매칭되지 않은 대여소 {info['unmatched']}곳은 ID로 표시합니다.")
    return df_pred, info


def station_row(df, station_id):
    """사이드바에서 고른 station_id에 해당하는 행(Series). 없으면 None."""
    if station_id is None:
        return None
    match = df[df["대여소ID"] == station_id]
    if match.empty:
        st.caption("선택한 대여소는 이번 조건에서 예측 결과가 없어 전체 보기로 표시합니다.")
        return None
    return match.iloc[0]


def make_base_map(df, selected=None):
    """전체 보기는 서울 전역, 대여소를 고르면 해당 위치로 확대 + 빨간 원으로 표시."""
    if selected is None:
        m = folium.Map(location=[df["위도"].mean(), df["경도"].mean()], zoom_start=12)
    else:
        m = folium.Map(location=[selected["위도"], selected["경도"]], zoom_start=16)
        folium.CircleMarker(
            location=[selected["위도"], selected["경도"]],
            radius=22,
            color="#e63946",
            weight=3,
            fill=False,
        ).add_to(m)
    return m


def map_key(prefix, selected):
    """선택이 바뀔 때마다 지도를 새로 그려서 확대 위치가 확실히 적용되도록 하는 key."""
    station = "all" if selected is None else selected["대여소ID"]
    return f"{prefix}_{model_type}_{selected_date}_{period_label}_{station}"


# [추가 2026-11-17] 흐름 예보 화면(운영자용). 처음엔 1번 메뉴로 만들었으나, 이용자에게는 '내 대여소에서 빌리고 반납할 수 있나'가
#        중요하다는 사용자 판단에 따라 2번 메뉴 '운영자 보기'의 한 화면으로 옮겼다(1번은 이전 화면으로 되돌림).
# [역할] 운영자가 '어디서 자전거가 많이 빠져나가고 들어오나'를 보고 재배치 우선순위를 정하게 한다.
#        우리 예측의 강점(대여·반납, 시간당 오차 약 1.4대)을 쓰고, 약점인 재고 추정(오차 약 8.7대)은 '참고'로만 둔다.
# [원리] 증감 = 시간대 3시간 동안 (예측 반납 − 예측 대여).
#        −면 자전거가 빠져나가는 곳(빨강), +면 들어오는 곳(파랑), |증감| < 1대면 비슷(회색).
#        지도 원 크기는 |증감|의 제곱근에 비례(큰 값 몇 개가 화면을 덮지 않게).
def render_flow_forecast(df_predicted, info):
    window_text = f"{day_text(selected_date)} {period_label} ({info['hour_min']:02d}~{info['hour_max']:02d}시)"
    st.markdown(f"#### 🔴🔵 대여·반납 흐름 예보 (재배치 담당자용) — {window_text}")
    st.caption("대여소마다 이 시간대에 자전거가 얼마나 빠져나가고 들어올지(예측 대여·반납) 보여 줍니다. 재배치 우선순위를 정할 때 참고하세요.")

    fc = df_predicted.copy()
    fc["증감"] = fc["순수요_NetFlow"]
    out_n, in_n = int((fc["증감"] <= -1).sum()), int((fc["증감"] >= 1).sum())
    # 4칸을 한 줄에 두면 좁은 화면에서 숫자가 잘려('약 1,…') 2칸씩 두 줄로 놓는다.
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    m1.metric("빌려 갈 자전거 (합계)", f"{fc['예측_대여'].sum():,.0f}대", help="대상 60곳에서 이 시간대에 빌려 갈 것으로 예측한 자전거 수")
    m2.metric("반납될 자전거 (합계)", f"{fc['예측_반납'].sum():,.0f}대", help="대상 60곳에 이 시간대에 반납될 것으로 예측한 자전거 수")
    m3.metric("🔴 빠져나가는 곳", f"{out_n}곳", help="(반납 − 대여)가 −1대 이하인 대여소")
    m4.metric("🔵 들어오는 곳", f"{in_n}곳", help="(반납 − 대여)가 +1대 이상인 대여소")

    # 많이 빠져나갈 곳 / 많이 들어올 곳 상위 5곳
    def top_list(df):
        return "\n".join(f"{i}. **{r['대여소명']}** ({r['클러스터']}) — **{r['증감']:+.0f}대** "
                         f"(빌려 감 {r['예측_대여']:.0f} · 반납 {r['예측_반납']:.0f})"
                         for i, (_, r) in enumerate(df.iterrows(), 1))
    c_out, c_in = st.columns(2)
    with c_out:
        st.markdown("##### 🔴 자전거가 많이 빠져나갈 곳")
        st.markdown(top_list(fc.nsmallest(5, "증감")) or "없음")
        st.caption("빌리려면 일찍 가거나 근처 대여소도 함께 확인하세요.")
    with c_in:
        st.markdown("##### 🔵 자전거가 많이 들어올 곳")
        st.markdown(top_list(fc.nlargest(5, "증감")) or "없음")
        st.caption("반납하려면 거치대가 찰 수 있어요(다른 자전거에 연결해 반납 가능).")

    # 지도: 증감 크기·방향
    st.markdown(
        "<span style='color:#d62728;font-size:1.2em'>●</span> 빠져나가는 곳 &nbsp; "
        "<span style='color:#1f77b4;font-size:1.2em'>●</span> 들어오는 곳 &nbsp; "
        "<span style='color:#999;font-size:1.2em'>●</span> 비슷함 &nbsp; "
        "<span style='color:#888'>· 원이 클수록 변화가 커요 · 원을 누르면 자세히 보여요</span>",
        unsafe_allow_html=True,
    )
    selected_station = station_row(df_predicted, selected_station_id)
    m = make_base_map(df_predicted, selected_station)
    for _, row in fc.iterrows():
        is_selected = selected_station is not None and row["대여소ID"] == selected_station["대여소ID"]
        net = float(row["증감"])
        color = "#d62728" if net <= -1 else "#1f77b4" if net >= 1 else "#999999"
        word = "빠져나감" if net <= -1 else "들어옴" if net >= 1 else "비슷함"
        popup_text = (
            f"<div style='width:240px'><b>{row['대여소명']}</b> ({row['클러스터']})<br>"
            f"• 빌려 갈 자전거 약 {row['예측_대여']:.0f}대 · 반납 약 {row['예측_반납']:.0f}대<br>"
            f"• <b>증감 {net:+.0f}대 ({word})</b><br>"
            f"<span style='color:#888'>참고(정확도 낮음): 끝날 때 남은 자전거 {inv_text(row['예측_잔여수량'], row['총거치대수'])}"
            f" · 거치대 {int(row['총거치대수'])}석 · {str(row['위험상태']).split(' (')[0]}"
            + (" · 상시 과밀" if row["상시과밀"] else "") + "</span></div>"
        )
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=float(np.clip(4 + 2.2 * np.sqrt(abs(net)), 4, 22)),
            popup=folium.Popup(popup_text, max_width=300, show=is_selected),
            tooltip=f"{row['대여소명']} · {net:+.0f}대 ({word})",
            color=color, weight=1, fill=True, fill_color=color, fill_opacity=0.65,
        ).add_to(m)
    st_folium(m, width="100%", height=450, returned_objects=[], key=map_key("map_flow", selected_station))

    # 대여소별 예보표 (재고 기반 상태는 '참고' 열로 뒤에 둔다)
    with st.expander("📋 대여소별 예보표 보기", expanded=False):
        cols = ["대여소명", "클러스터", "예측_대여", "예측_반납", "증감"] + (["실제_NetFlow"] if info["has_actual"] else []) + ["위험상태"]
        table = fc[cols].sort_values("증감").rename(columns={
            "대여소명": "대여소", "클러스터": "자치구", "예측_대여": "빌려 갈 자전거", "예측_반납": "반납될 자전거",
            "증감": "증감 (+들어옴/−빠져나감)", "실제_NetFlow": "참조일 실제 증감", "위험상태": "참고: 재고 상태(추정)"})
        st.dataframe(one_based(table), width="stretch", height=400, column_config={
            "빌려 갈 자전거": st.column_config.NumberColumn(format="%.0f대"),
            "반납될 자전거": st.column_config.NumberColumn(format="%.0f대"),
            "증감 (+들어옴/−빠져나감)": st.column_config.NumberColumn(format="%+.1f대",
                                                               help="시간대 동안 (예측 반납 − 예측 대여)"),
            "참조일 실제 증감": st.column_config.NumberColumn(format="%+.1f대"),
            "참고: 재고 상태(추정)": st.column_config.TextColumn(
                help="작년 같은 시각 평균 재고로 추정한 값이라 정확도가 낮아요(오차 약 8.7대). 참고로만 보세요."),
        })
        st.caption("빠져나가는 순서로 정렬했어요. 재고 상태(부족·거치대 거의 참)는 작년 평균 재고로 추정한 값이라 참고로만 보세요.")
        if info["has_actual"]:
            st.caption(f"참조일 실제 증감: 참조일({info['ref_date']}) 같은 시간대의 실제 값으로, 예측과 비교하는 용도입니다.")


# [추가 2026-11-17 이동 흐름 = 서비스 목표 3] 고른 대여소에서 빌린 자전거가 어디로 가는지(또는 어디서 오는지) 화살표로 보여 준다.
# [역할] 운영자가 대여소 하나를 기준으로 "자전거가 어느 방향으로 빠져나가고 어디서 들어오는지" 보고 회수·보충 경로를 정하게 한다.
# [원리] scripts/build_od_shares.py가 2025년 이동 기록으로 만든 '상위 10곳 비율표'(평일/주말·공휴일 × 출근/퇴근)를 읽고,
#        예상 대수 = 비율 × 그날 이 대여소의 예측 대여 수(나가는 흐름) 또는 예측 반납 수(들어오는 흐름).
#        검증(2024 비율 → 2025): 상위 10곳 중 약 7곳이 그대로였고, 경로별 대수 오차 약 25~32%(상관 0.95) → '방향을 보는' 용도.
#        60곳에서 빌린 자전거의 90% 이상이 60곳 밖으로 가므로, 서울 전체 대여소 좌표로 상대를 그린다(60곳 밖은 예측 없이 위치만).
#        권역(구) 단위는 재배치에 쓰기엔 너무 넓다는 사용자 판단에 따라 대여소 단위로 만든다. 날씨는 반영하지 않는다(검증 안 함).
OD_FILE = DATA_DIR / "od_top_partners.csv"


@st.cache_data(show_spinner=False)
def load_od():
    if not OD_FILE.exists():
        return None
    od = pd.read_csv(OD_FILE, encoding="utf-8-sig", dtype={"station_id": str, "partner_id": str})
    return od


def render_od_flow(df_predicted, info):
    od = load_od()
    window_text = f"{day_text(selected_date)} {period_label} ({info['hour_min']:02d}~{info['hour_max']:02d}시)"
    st.markdown(f"#### 🔀 이동 흐름 (재배치 담당자용) — {window_text}")
    if od is None:
        st.info("이동 비율표(data/od_top_partners.csv)가 없습니다. scripts/build_od_shares.py로 만들어 주세요.")
        return
    names = df_predicted.set_index("대여소ID")["대여소명"]
    ids = sorted(names.index, key=lambda s: names[s])
    default = ids.index(selected_station_id) if selected_station_id in ids else 0
    c1, c2, c3 = st.columns([2, 2, 1.4])
    sid = c1.selectbox("기준 대여소", ids, index=default, format_func=lambda s: names[s], key="od_station",
                       help="사이드바에서 대여소를 고르면 그 대여소가 기본으로 선택돼요.")
    dir_label = c2.segmented_control("흐름", ["나가는 흐름", "들어오는 흐름"], default="나가는 흐름", key="od_dir") or "나가는 흐름"
    # [추가 2026-11-17] 표시 개수 조절: 화살표가 많아 복잡할 때 상위 3·5곳만 보게 한다.
    top_n = int((c3.segmented_control("표시", ["3곳", "5곳", "10곳"], default="10곳", key="od_topn") or "10곳").rstrip("곳"))
    out = dir_label == "나가는 흐름"
    kind = "off" if (selected_date.weekday() >= 5 or selected_date.strftime("%Y-%m-%d") in load_holiday_set()) else "work"
    rows = od[(od.station_id == str(sid)) & (od.direction == ("out" if out else "in"))
              & (od.period == PERIOD_OPTIONS[period_label]) & (od.kind == kind)].sort_values("rank")
    covered_all = float(rows["share"].sum())   # 상위 10곳 전체 비중(표시 개수와 관계없이 안내용)
    rows = rows[rows["rank"] <= top_n]
    me = df_predicted[df_predicted["대여소ID"] == sid].iloc[0]
    if rows.empty:
        st.caption("이 조건의 이동 기록이 없어요.")
        return
    base = float(me["예측_대여"] if out else me["예측_반납"])
    rows = rows.assign(예상=rows["share"] * base,
                       거리_m=haversine_m(me["위도"], me["경도"], rows["partner_lat"].to_numpy(float), rows["partner_lon"].to_numpy(float)))
    # [수정 2026-11-17 식별성] 화살표가 많으면 한 색으로는 구분이 어려워, 비율 단계별 색(진함 → 옅음) + 도착지 순위 번호(표와 같은 번호)로 구분한다.
    # [원리] 비율 15% 이상 = 주요 흐름(진한 색), 5~15% = 중간, 5% 미만 = 옅은 회색. 나가는 흐름은 빨강 계열, 들어오는 흐름은 파랑 계열.
    tiers = ([("15% 이상", 0.15, "#a50f15"), ("5~15%", 0.05, "#fb6a4a"), ("5% 미만", 0.0, "#bdbdbd")] if out else
             [("15% 이상", 0.15, "#08519c"), ("5~15%", 0.05, "#6baed6"), ("5% 미만", 0.0, "#bdbdbd")])

    def tier_of(share):
        return next(t for t in tiers if share >= t[1])

    st.caption(f"{'빌려 갈' if out else '반납될'} 자전거 예측 약 {base:.0f}대 중, "
               f"{'어디로 가는지' if out else '어디서 오는지'} 작년({int(rows['base_year'].iloc[0])}년) "
               f"{'평일' if kind == 'work' else '주말·공휴일'} 같은 시간대 비율로 나눴어요.")

    # 범례: 비율 단계 색 + 번호 설명
    st.markdown(" &nbsp; ".join(f"<span style='color:{c};font-size:1.3em'>━</span> {label}" for label, _, c in tiers)
                + " &nbsp; <span style='color:#888'>· 굵기 = 오늘 예상 대수 · 동그라미 번호 = 아래 표의 순위</span>",
                unsafe_allow_html=True)
    # 지도: 기준 대여소 ↔ 상위 N곳 화살표. 굵기 = 예상 대수(가장 큰 것 기준 2~12px), 색 = 비율 단계, '►'로 방향 표시.
    # 옅은 선을 먼저, 진한 선을 나중에 그려 주요 흐름이 위에 보이게 한다.
    pts = [[me["위도"], me["경도"]]] + rows[["partner_lat", "partner_lon"]].values.tolist()
    center, zoom = fit_view(pts, padding_px=90)   # 여백을 넉넉히: 기준 대여소(★)가 지도 가장자리에 잘리지 않게
    m = folium.Map(location=center, zoom_start=zoom)
    top = max(float(rows["예상"].max()), 1e-9)
    for _, r in rows.sort_values("share").iterrows():
        a, b = [me["위도"], me["경도"]], [r["partner_lat"], r["partner_lon"]]
        c = tier_of(float(r["share"]))[2]
        line = folium.PolyLine([a, b] if out else [b, a], color=c, weight=2 + 10 * float(r["예상"]) / top, opacity=0.85,
                               tooltip=f"{r['rank']}위 {r['partner_name']} · 약 {r['예상']:.1f}대 ({r['share'] * 100:.0f}%)")
        line.add_to(m)
        PolyLineTextPath(line, "  ►  ", repeat=True, offset=5, attributes={"fill": c, "font-size": "12"}).add_to(m)
        # 도착(출발) 대여소 위치에 순위 번호 동그라미(표의 '순위'와 같음). 60곳 대상은 테두리를 굵게.
        border = "3px solid #222" if r["partner_in_60"] else "1px solid #fff"
        folium.Marker(b, tooltip=f"{r['rank']}위 {r['partner_name']}{' (분석 대상 60곳)' if r['partner_in_60'] else ''}",
                      icon=folium.DivIcon(icon_size=(22, 22), icon_anchor=(11, 11), html=(
                          f"<div style='width:22px;height:22px;border-radius:11px;background:{c};border:{border};"
                          f"color:#fff;font-weight:700;font-size:12px;line-height:20px;text-align:center;"
                          f"box-shadow:0 0 2px #000'>{int(r['rank'])}</div>"))).add_to(m)
    folium.Marker([me["위도"], me["경도"]], tooltip=f"기준: {me['대여소명']}",
                  icon=folium.Icon(color="black", icon="star", prefix="glyphicon")).add_to(m)
    st_folium(m, width="100%", height=430, returned_objects=[],
              key=f"od_map_{sid}_{dir_label}_{top_n}_{selected_date}_{period_label}")

    covered = covered_all
    # 표에 비율 단계를 같은 색 이름으로 보여 줘 지도 색과 맞춰 볼 수 있게 한다.
    dots = {t[0]: d for t, d in zip(tiers, ["🟥" if out else "🟦", "🟧" if out else "🔹", "⬜"])}
    table = rows.assign(대상=np.where(rows["partner_in_60"], "60곳", "-"),
                        단계=[f"{dots[tier_of(float(s))[0]]} {tier_of(float(s))[0]}" for s in rows["share"]])[
        ["rank", "partner_name", "단계", "대상", "예상", "share", "per_day", "거리_m"]].rename(columns={
        "rank": "순위", "partner_name": "도착 대여소" if out else "출발 대여소", "예상": "오늘 예상",
        "share": "비율", "per_day": "작년 하루 평균", "거리_m": "거리(직선)"})
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "오늘 예상": st.column_config.NumberColumn(format="%.1f대", help="비율 × 그날 이 대여소의 예측 대여(반납) 수"),
        "비율": st.column_config.NumberColumn(format="percent", help="작년 같은 조건에서 전체 중 이 대여소로(에서) 간 비율"),
        "작년 하루 평균": st.column_config.NumberColumn(format="%.1f대"),
        "거리(직선)": st.column_config.NumberColumn(format="%.0fm"),
        "대상": st.column_config.TextColumn(help="60곳 = 분석 대상(예측 있음), - = 60곳 밖(위치만, 예측 없음)"),
    })
    st.caption(f"상위 10곳이 전체의 약 {covered * 100:.0f}%예요. 나머지 {100 - covered * 100:.0f}%는 여러 곳으로 조금씩 흩어져요. "
               "작년 비율로 다음 해를 맞혀 보면 상위 10곳 중 약 7곳이 그대로였고 경로별 대수는 25~32% 정도 틀려, "
               "정확한 대수보다 '흐르는 방향'을 보는 용도예요. 화살표는 직선이고, 날씨는 반영하지 않아요.")


# [추가 2026-11-18 '모자란 순' 정렬 수정]
# [역할] 1번 메뉴 표의 '모자란 순'에서 상태가 '부족 위험'인 대여소를 맨 위에 놓는다.
# [왜] 예전에는 증감(반납 − 대여)만으로 정렬해, 많이 빠져나가도 원래 자전거가 많은 '정상' 대여소가 맨 위에 왔다.
#      다음 7일 예측으로 세어 보니 맨 위 5곳 중 81%가 '부족'이 아니어서, 이름(모자란 순)과 표 내용이 어긋났다.
# [원리] 1순위 = 부족 위험 여부(부족이 먼저), 2순위 = 증감이 작은(많이 빠져나가는) 순. 부족 대여소끼리도 증감 순으로 놓인다.
def shortage_first(df):
    key = (~df["위험상태"].str.startswith("부족")).astype(int)
    return df.assign(_short_key=key).sort_values(["_short_key", "순수요_NetFlow"]).drop(columns="_short_key")


NEAR_WARN_M = 2000   # 이 거리 안에 대상 대여소가 하나도 없으면 '멀다'고 알린다


def nearest_sort(df):
    """[역할] 주소·장소 입력칸을 그리고, 그 위치에서 가까운 순으로 정렬한 표(거리_m 열 추가)를 돌려준다.
    [원리] 버튼을 눌렀을 때만 카카오를 부른다(입력할 때마다 호출하지 않게). 키가 없거나 검색에 실패하면 원래 표를 그대로 돌려준다."""
    if not kakao_key():
        st.info("가까운 순 정렬에는 카카오 API 키(KAKAO_REST_KEY)가 필요해요. 키가 없어 원래 순서로 보여 드려요.")
        return df
    with st.form("near_form", border=False):
        # key를 고정해야 한다: 기본값(value)을 바꿔 가며 넘기면 입력칸이 새로 만들어져 방금 입력한 글자가 사라진다.
        q = st.text_input("📍 기준 위치", placeholder="주소나 장소 이름 (예: 천호역, 서울 강동구 천호대로 1005)",
                          key="near_input")
        if st.form_submit_button("가까운 순으로 보기"):
            st.session_state["near_query"] = q.strip()
    query = st.session_state.get("near_query", "")
    if not query:
        st.caption("기준 위치를 입력하고 '가까운 순으로 보기'를 눌러 주세요.")
        return df
    place = pick_place("기준 위치", query, st)
    if place is None:
        return df
    out = df.assign(거리_m=haversine_m(place["lat"], place["lon"],
                                      df["위도"].to_numpy(dtype=float), df["경도"].to_numpy(dtype=float)))
    out = out.sort_values("거리_m")
    if not out.empty and out["거리_m"].iloc[0] > NEAR_WARN_M:
        st.warning(f"'{place['name']}' 근처 2km 안에는 분석 대상 대여소가 없어요. 가장 가까운 곳도 약 "
                   f"{fmt_distance(out['거리_m'].iloc[0])} 떨어져 있어요(분석 대상 60곳 기준).")
    else:
        st.caption(f"'{place['name']}'에서 직선거리로 가까운 순이에요. 실제 길은 더 길 수 있어요.")
    return out


# [수정 2026-11-17] 4번 메뉴 안에 있던 함수를 밖으로 옮겼다. 1번 메뉴의 '가까운 순' 정렬에서도 같이 쓰기 위해서다(동작은 그대로).
def pick_place(label, query, col):
    """카카오 검색 → 결과가 여러 개면 고르게 한다. 실패하면 이유를 보여 주고 None."""
    try:
        results = kakao_search(query)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code in (401, 403):
            col.error(
                f"카카오 API 키가 올바르지 않거나 사용 권한이 없습니다(오류 {code}). "
                "secrets.toml의 키와 카카오 개발자 사이트의 앱 설정을 확인해 주세요."
            )
        else:
            col.warning(f"'{query}' 검색에 실패했습니다(오류 {code}). 잠시 후 다시 시도해 주세요.")
        return None
    except Exception:
        col.warning(f"'{query}' 검색 중 카카오 서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.")
        return None
    if not results:
        col.warning(f"'{query}' 검색 결과가 없습니다. 다른 이름이나 주소로 검색해 주세요.")
        return None
    labels = [f"{r['name']} · {r['address']}" if r["address"] else r["name"] for r in results]
    idx = col.selectbox(
        f"{label} 검색 결과", range(len(results)), format_func=lambda i: labels[i],
        key=f"pick_{label}_{query}",
    )
    return results[idx]


# ==========================================
# 5. 메인 화면 분기
# ==========================================

# ------------------------------------------
# Mode 1: 실시간 지도 시각화
# ------------------------------------------
if mode == MENU1:   # [수정 2026-11-17] 출퇴근 예보 시도 후 사용자 결정으로 이전 화면으로 되돌림
    df_predicted, info = get_predictions()
    window_text = f"{day_text(selected_date)} {period_label} ({info['hour_min']:02d}~{info['hour_max']:02d}시)"
    # [수정 2026-11-17] Net-flow 등 전문 용어를 쉬운 말로, 모델 이름은 끝 괄호로
    with st.container(key="subheader_box_1"):
        st.subheader(f"📊 {window_text} 대여소별 자전거 남음·부족 예측 · {model_title}")

    shortage_cnt = len(df_predicted[df_predicted["위험상태"].str.contains("부족")])
    overflow_cnt = len(df_predicted[df_predicted["위험상태"].str.contains("포화")])

    st.markdown("---")
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # [수정 2026-11-17] 제목과 숫자 3개를 한 줄에 나란히 두면 좁은 화면(약 800px)에서 서로 겹쳤다 → 위아래로 쌓는다.
        title_col = stats_col = st.container()
        title_col.markdown(
            '<div class="compact-heading">🗺️ 대여소 현황 지도 (예측)</div>',
            unsafe_allow_html=True,
        )
        # 값과 변화 태그를 한 줄에 놓기 위해 st.metric 대신 직접 만든 HTML을 사용 (줄바꿈 없이 한 줄로 조립)
        stats_html = (
            '<div class="stat-row">'
            '<div class="stat"><div class="stat-label">대상 대여소</div>'
            f'<div class="stat-line"><span class="stat-value">{len(df_predicted)}곳</span></div></div>'
            '<div class="stat"><div class="stat-label">자전거 부족 예상 '
            # [수정 2026-11-18] 부족 판정 보강(예측 또는 작년 기록)에 맞춰 설명을 바꿈
            '<span class="stat-help" title="끝날 때 남은 자전거가 3대 이하로 예측되거나, 작년 이 시기 절반 이상의 날에 3대 이하였던 대여소">?</span></div>'
            f'<div class="stat-line"><span class="stat-value">{shortage_cnt}곳</span>'
            '</div></div>'   # [수정 2026-11-17] 뜻이 모호한 '↓ -경고' 태그 제거
            '<div class="stat"><div class="stat-label">거치대 거의 참 '
            '<span class="stat-help" title="예측 잔여 자전거가 거치대 수 - 3대 이상이라 거치대가 거의 찬 대여소. 반납은 가능하지만 다른 자전거에 연결해 반납해야 할 수 있음. 평소에도 거치대보다 자전거가 많은 상시 과밀 대여소가 포함됨">?</span></div>'
            f'<div class="stat-line"><span class="stat-value">{overflow_cnt}곳</span>'
            '</div></div>'   # [수정 2026-11-17] '↑ +주의' 태그 제거
            '</div>'
        )
        stats_col.markdown(stats_html, unsafe_allow_html=True)
        # [추가 2026-11-17] 지도 표시 색 설명(범례). [원리] 아래 마커 색 규칙(부족=빨강, 포화=파랑, 그 밖=초록)과 같게 적는다.
        st.markdown(
            "<span style='color:#d63e2a;font-size:1.2em'>●</span> 자전거 부족 예상 &nbsp; "
            "<span style='color:#38aadd;font-size:1.2em'>●</span> 거치대 거의 참 (연결 반납 필요할 수 있음) &nbsp; "
            "<span style='color:#72b026;font-size:1.2em'>●</span> 보통 &nbsp; "
            "<span style='color:#888'>· 표시를 누르면 자세한 예측이 나와요</span>",
            unsafe_allow_html=True,
        )
        selected_station = station_row(df_predicted, selected_station_id)
        m = make_base_map(df_predicted, selected_station)

        for _, row in df_predicted.iterrows():
            is_selected = (
                selected_station is not None and row["대여소ID"] == selected_station["대여소ID"]
            )
            if "부족" in row["위험상태"]:
                color = "red"
                icon_type = "exclamation-sign"
            elif "포화" in row["위험상태"]:
                color = "blue"
                icon_type = "info-sign"
            else:
                color = "green"
                icon_type = "ok-sign"

            popup_text = (
                f"<div style='width:230px'>"
                f"<b>{row['대여소명']}</b><br>"
                f"• 지역: {row['클러스터']} ({row['구분']})<br>"
                f"• 상태: <b>{row['위험상태']}</b><br>"
                f"• 예측 대여 {row['예측_대여']:.1f}대 / 반납 {row['예측_반납']:.1f}대<br>"
                f"• 증감: {row['순수요_NetFlow']:+.1f}대 ({'남음' if row['순수요_NetFlow'] >= 0 else '모자람'})<br>"
                # [수정 2026-11-17] 예측 잔여가 거치대 수를 넘을 수 있고, 평소에도 넘는 '상시 과밀' 대여소는 따로 알린다.
                f"• <b>예측 잔여: {inv_text(row['예측_잔여수량'], row['총거치대수'])}</b> (거치대 {int(row['총거치대수'])}석, 추정)"
                # [추가 2026-11-18] 작년 기록 때문에 부족이 된 경우 그 이유를 적는다(예측 잔여는 넉넉해 보여도 부족일 수 있으므로)
                + ("<br>• 작년 이 시기엔 절반 이상의 날에 3대 이하였어요" if "작년" in str(row["부족근거"]) else "")
                + (f"<br>• 상시 과밀: 평소에도 약 {int(row['평소재고'])}대가 놓여 있는 곳" if row["상시과밀"] else "")
                + "</div>"
            )

            folium.Marker(
                location=[row["위도"], row["경도"]],
                popup=folium.Popup(popup_text, max_width=300, show=is_selected),
                # [추가 2026-11-17] 누르지 않아도 마우스를 올리면 이름·상태가 보이게 한다
                tooltip=f"{row['대여소명']} · {str(row['위험상태']).split(' (')[0]}",
                icon=folium.Icon(color=color, icon=icon_type, prefix="glyphicon"),
            ).add_to(m)

        st_folium(
            m,
            width="100%",
            height=450,
            returned_objects=[],
            key=map_key("map1", selected_station),
        )

    with right_col:
        st.markdown(
            '<div class="compact-heading">📋 대여소별 예측표</div>', unsafe_allow_html=True
        )
        table_cols = ["대여소명", "순수요_NetFlow", "위험상태"]
        if info["has_actual"]:
            table_cols.insert(2, "실제_NetFlow")
        # [추가 2026-11-17] 검색칸: 이름·옛 이름·번호로 표를 좁혀 본다(지도는 그대로 전체)
        base = station_search(df_predicted, key="search_m1")
        # [추가 2026-11-17 정렬·가까운 순] 표 순서를 고른다. '가까운 순'은 주소·장소를 검색해 그 위치에서 직선거리 순으로 놓는다.
        # [원리] 4번 메뉴와 같은 카카오 검색(pick_place)으로 위치를 찾고, haversine_m으로 대여소까지 직선거리를 잰다.
        #        GPS 대신 주소를 쓰는 이유: 위치 권한 팝업·PC 위치 부정확·새 부품 설치 없이 동작한다.
        #        대상이 60곳뿐이라 가까운 곳이 멀 수 있어, 2km 안에 없으면 그렇다고 알려 준다.
        sort_by = st.segmented_control("표 정렬", ["모자란 순", "가까운 순", "이름순"], default="모자란 순",
                                    key="sort_m1", help="모자란 순: '자전거 부족 예상' 대여소를 맨 위에, 그다음 자전거가 많이 빠져나가는 순")
        if sort_by == "가까운 순":
            base = nearest_sort(base)
            if "거리_m" in base.columns:
                table_cols.insert(1, "거리_m")
            else:   # 위치를 못 찾았으면 기본 순서(모자란 순)로 보여 준다
                base = shortage_first(base)
        elif sort_by == "이름순":
            base = base.sort_values("대여소명")
        else:   # 모자란 순(기본, 버튼을 다시 눌러 선택을 지운 경우 포함)
            base = shortage_first(base)
        table = base[table_cols].copy()
        for c in ("순수요_NetFlow", "실제_NetFlow"):
            if c in table.columns:
                table[c] = table[c].round(1)
        # [수정 2026-11-17] 열 이름을 쉬운 말로, 숫자는 그대로 두고 '+/−대' 형식으로 표시(머리글 정렬 유지)
        net_col, act_col = "증감 (+남음/−모자람)", "참조일 실제 증감"
        table = table.rename(columns={"대여소명": "대여소", "순수요_NetFlow": net_col,
                                    "실제_NetFlow": act_col, "위험상태": "상태", "거리_m": "거리(직선)"})
        st.dataframe(one_based(table), width="stretch", height=400, column_config={
            "거리(직선)": st.column_config.NumberColumn(format="%.0fm", help="고른 위치에서 대여소까지 직선거리(실제 길은 더 길어요)"),
            net_col: st.column_config.NumberColumn(format="%+.1f대",
                                                help="시간대 동안 (예측 반납 − 예측 대여). +면 자전거가 남고, −면 모자랍니다."),
            act_col: st.column_config.NumberColumn(format="%+.1f대"),
        })
        st.caption("증감: 이 시간대에 들어오는 자전거(반납)에서 나가는 자전거(대여)를 뺀 값이에요.")
        if info["has_actual"]:
            st.caption(
                f"참조일 실제 증감: 참조일({info['ref_date']}) 같은 시간대의 실제 값으로, "
                "예측이 그날을 얼마나 맞혔는지 비교하는 용도입니다."
            )

# ------------------------------------------
# Mode 2: 대여소 유형 지도 (K-Means)
# ------------------------------------------
# [수정 2026-11-17 1단계] 이전 화면은 이름만 K-Means였고 실제로는 자치구별 묶음이었다(클러스터 = 구).
# [역할] 비전공자가 '내 대여소가 어떤 곳인지, 오늘 그 유형이 어떨지'를 바로 알 수 있게 한다.
#        ① 제목·오늘 요약 ③ 유형별 색 지도(+자치구 필터) ④ 유형 설명 카드 ⑥ 분석 방법(작은 글씨) 순서.
# [원리] 유형은 2025년 평일 기록으로 미리 계산한 K-Means 결과(load_station_types)이고,
#        '오늘' 값은 사이드바에서 고른 날짜·시간대의 예측(get_predictions)을 유형별로 모은 것이다.
elif mode == "2. 대여소 유형 지도 (출퇴근 흐름별)":
    df_predicted, info = get_predictions()
    types, tmeta = load_station_types()
    with st.container(key="subheader_box_2"):
        st.subheader("🧭 대여소 유형 지도 (출퇴근 흐름별) — 비슷하게 움직이는 대여소끼리 묶었어요")
    if types is None:
        st.error("대여소 유형 파일(data/station_types.csv)이 없습니다. "
                 "`.venv\\Scripts\\python.exe scripts\\build_station_types.py`를 실행해 만들어 주세요.")
        st.stop()
    st.markdown(
        "평일 출퇴근 때 자전거가 **들어오는 곳인지, 빠져나가는 곳인지**에 따라 대여소를 "
        f"{len(tmeta['types'])}가지 유형으로 나눴어요. 지도의 **색깔은 유형**, **굵은 테두리는 오늘 예측 위험**입니다."
    )
    d = df_predicted.merge(types, left_on="대여소ID", right_on="station_id", how="left")
    period_word = "출근" if PERIOD_OPTIONS[period_label] == "morning" else "퇴근"

    # ① 오늘 요약 — [원리] 유형별 대여소당 평균 (반납-대여)와 부족·포화 위험 대여소 수를 센다. +면 남음, -면 모자람.
    def type_today(key):
        g = d[d["type_key"] == key]
        if g.empty:
            return None
        return {"net": float(g["순수요_NetFlow"].mean()),
                "short": int(g["위험상태"].str.startswith("부족").sum()),
                "full": int(g["위험상태"].str.startswith("포화").sum()), "n": len(g),
                # [추가 2026-11-17] 포화 중 '상시 과밀'(평소에도 거치대 초과) 대여소 수 → 오늘만의 위험과 구분
                "full_chronic": int((g["위험상태"].str.startswith("포화") & g["상시과밀"]).sum()),
                "chronic": int(g["상시과밀"].sum())}

    def net_text(net):
        if abs(net) < 0.5:
            return "들고 나는 양이 비슷"
        return f"대여소당 평균 {abs(net):.1f}대 {'남음' if net > 0 else '모자람'}"

    summary_lines = []
    for t in tmeta["types"]:
        s = type_today(t["key"])
        if s:
            risk = " · ".join(x for x in [f"부족 위험 {s['short']}곳" if s["short"] else "",
                                         (f"포화(거치대 거의 참) {s['full']}곳"
                                          + (f" — 그중 상시 과밀 {s['full_chronic']}곳" if s["full_chronic"] else ""))
                                         if s["full"] else ""] if x) or "위험 없음"
            summary_lines.append(f"- {t['emoji']} **{t['name']}** ({s['n']}곳): {net_text(s['net'])} · {risk}")
    st.info(f"**{day_text(selected_date)} {period_label} 예측 요약** "
            f"(남음 = 반납이 대여보다 많음)\n\n" + "\n".join(summary_lines)
            + "\n\n포화는 반납이 안 된다는 뜻이 아니라, 거치대가 거의 차서 다른 자전거에 연결해 반납해야 할 수 있다는 뜻이에요. "
            "**상시 과밀**은 평소에도 거치대보다 자전거가 많은 곳이라 매일 그럴 수 있어요.")

    # ③ 유형별 색 지도 + 자치구 필터 — [원리] 채움색 = 유형 색, 테두리 = 부족(빨강)·포화(남색)일 때만 굵게.
    gu_options = ["전체"] + sorted(d["클러스터"].dropna().unique().tolist())
    gu_pick = st.selectbox("자치구로 좁혀 보기", gu_options, key="type_gu")
    view = d if gu_pick == "전체" else d[d["클러스터"] == gu_pick]
    legend = " &nbsp; ".join(
        f"<span style='color:{t['color']};font-size:1.3em'>●</span> {t['emoji']} {t['name']}" for t in tmeta["types"])
    st.markdown(legend + " &nbsp; | &nbsp; <b style='color:#e63946'>◯ 부족 위험</b> &nbsp; "
                "<b style='color:#1d3557'>◯ 포화(거치대 거의 참)</b>", unsafe_allow_html=True)

    selected_station = station_row(df_predicted, selected_station_id)
    if selected_station is None and gu_pick != "전체":
        center, zoom = fit_view(view[["위도", "경도"]].values.tolist())   # 고른 구의 대여소가 모두 보이게
        m_type = folium.Map(location=center, zoom_start=zoom)
    else:
        m_type = make_base_map(df_predicted, selected_station)
    for _, row in view.iterrows():
        is_selected = selected_station is not None and row["대여소ID"] == selected_station["대여소ID"]
        fill = row["type_color"] if pd.notna(row["type_color"]) else "#999999"
        risk = str(row["위험상태"])
        edge = "#e63946" if risk.startswith("부족") else "#1d3557" if risk.startswith("포화") else fill
        net = row["순수요_NetFlow"]
        popup = (f"<b>{row['대여소명']}</b> ({row['클러스터']})<br>"
                 f"유형: {row['type_emoji'] if pd.notna(row['type_emoji']) else ''} {row['type_name'] if pd.notna(row['type_name']) else '미분류'}<br>"
                 f"오늘 {period_word}: {'+' if net >= 0 else ''}{net:.0f}대 ({'남음' if net >= 0 else '모자람'})<br>{risk}"
                 + (f"<br>상시 과밀: 평소에도 약 {int(row['평소재고'])}대 (거치대 {int(row['총거치대수'])}석)"
                    if row["상시과밀"] else ""))
        folium.CircleMarker(
            location=[row["위도"], row["경도"]], radius=9,
            popup=folium.Popup(popup, max_width=300, show=is_selected),
            # [수정 2026-11-17] 마우스를 올리면 이름·유형·상태를 글로 보여 준다(색만으로 구분하기 어려운 이용자 배려)
            tooltip=f"{row['대여소명']} · {row['type_name'] if pd.notna(row['type_name']) else '미분류'} · {risk.split(' (')[0]}",
            color=edge, weight=4 if edge != fill else 1,
            fill=True, fill_color=fill, fill_opacity=0.85,
        ).add_to(m_type)
    st_folium(m_type, width="100%", height=450, returned_objects=[],
              key=map_key(f"map2_{gu_pick}", selected_station))

    # ② 내 대여소 카드 (2단계, 이용자용)
    # [수정 2026-11-17] 카드가 지도 위에 있으면 대여소를 고를 때마다 지도가 아래로 밀려, 지도 바로 아래로 옮겼다.
    # [역할] 사이드바에서 고른 대여소 하나에 대해 '어떤 곳인지 + 오늘 어떨지 + 그럼 어떻게 할지'를 한 장에 보여 준다.
    # [원리] 유형·평소 모습은 유형표(2025년 평일), 오늘 값은 예측표. 위험이면 같은 예측표에서 직선거리로 가까운
    #        '위험하지 않은' 대여소를 찾아 대안으로 제안한다(1.5km 이내, 실제 길찾기는 메뉴 4).
    my = d[d["대여소ID"] == selected_station_id] if selected_station_id is not None else d.iloc[0:0]
    if my.empty:
        st.caption("👉 사이드바 **대여소 선택**에서 대여소를 고르면 '내 대여소 카드'가 여기에 나타납니다.")
    else:
        r = my.iloc[0]
        with st.container(border=True):
            st.markdown(f"#### 📍 내 대여소: {r['대여소명']} ({r['클러스터']})")
            c1, c2 = st.columns([1, 1])
            with c1:
                if pd.notna(r["type_key"]):
                    st.markdown(f"**유형:** {r['type_emoji']} {r['type_name']}")
                    st.markdown(f"평소 출근: {flow_text(r['am_return_per_rental'])}  \n"
                                f"평소 퇴근: {flow_text(r['pm_return_per_rental'])}")
                if r["상시과밀"]:
                    st.caption(f"상시 과밀: 평소에도 약 {int(r['평소재고'])}대가 놓여 있어요(거치대 {int(r['총거치대수'])}석).")
            with c2:
                st.markdown(f"**오늘 {period_label} 예측** ({selected_date:%m/%d})  \n"
                            f"빌려 가는 자전거 약 {r['예측_대여']:.0f}대 · 반납 약 {r['예측_반납']:.0f}대  \n"
                            f"끝날 때 남은 자전거: **{inv_text(r['예측_잔여수량'], r['총거치대수'])}** (거치대 {int(r['총거치대수'])}석, 추정)  \n"
                            f"상태: **{r['위험상태']}**"
                            # [추가 2026-11-18] 부족 근거 표시: 작년 기록 때문이면 그 빈도를 쉬운 말로
                            + (f"  \n<small>작년 이 시기 10번 중 {round(float(r['부족_작년빈도']) * 10)}번은 "
                               f"시간대가 끝날 때 3대 이하였어요</small>"
                               if "작년" in str(r["부족근거"]) else ""), unsafe_allow_html=True)
            # [추가 2026-11-17] 고른 출퇴근 시간대의 3개 시각별 빌리기·반납하기 가능성(작년 실제 기록 기준)
            use_red = show_use_forecast(r["대여소ID"], selected_date, list(range(info["hour_min"], info["hour_max"] + 1)))
            risk_now = str(r["위험상태"])
            if risk_now.startswith("부족") or risk_now.startswith("포화"):
                bad = "부족" if risk_now.startswith("부족") else "포화"
                alt = d[(d["대여소ID"] != r["대여소ID"]) & ~d["위험상태"].str.startswith(bad)].copy()
                alt["거리"] = haversine_m(r["위도"], r["경도"], alt["위도"].to_numpy(float), alt["경도"].to_numpy(float))
                alt = alt[alt["거리"] <= 1500].nsmallest(2, "거리")
                what = "빌릴 자전거가 모자랄 수 있어요" if bad == "부족" else "거치대가 거의 차서 다른 자전거에 연결해 반납해야 할 수 있어요"
                # 대안 설명은 목적에 맞게: 빌릴 때는 남을 자전거 수, 반납할 때는 '거치대 여유'(자전거가 적을수록 자리가 많음)
                def alt_note(a):
                    # [수정 2026-11-17] 거치대를 넘는 예측은 숫자 대신 '거치대 초과'(inv_text)
                    return (f"남을 자전거 {inv_text(a['예측_잔여수량'], a['총거치대수'])}" if bad == "부족"
                            else f"거치대 여유 있음, 거치대 {int(a['총거치대수'])}석")
                tip = (" 가까운 대안: " + ", ".join(f"**{a['대여소명']}**({fmt_distance(a['거리'])}, {alt_note(a)})"
                                                  for _, a in alt.iterrows())) if len(alt) else " 1.5km 안에 대안 대여소가 없어요(분석 대상 60곳 기준)."
                st.warning(f"💡 {what}.{tip} 길찾기는 메뉴 4에서 할 수 있어요.")
            # [수정 2026-11-18 카드 안내 일치]
            # [역할] 예측 상태는 '정상'이어도, 위 시간대 표에 🔴가 있으면 '큰 문제 없음' 대신 그 시각을 알려 준다.
            # [원리] 예측 상태(예측 재고 기준)와 시간대 표(작년 실제 기록 기준)는 근거가 달라 엇갈릴 수 있다.
            #        둘 중 하나라도 어렵다고 하면 '주의'로 안내하고, 둘 다 괜찮을 때만 '큰 문제 없음'으로 안내한다.
            elif use_red and (use_red["rent"] or use_red["return"]):
                parts = []
                if use_red["rent"]:
                    parts.append(f"{'·'.join(f'{h}시' for h in use_red['rent'])}대에는 자전거가 거의 없던 날이 많았어요")
                if use_red["return"]:
                    parts.append(f"{'·'.join(f'{h}시' for h in use_red['return'])}대에는 반납 자전거가 많이 쌓였던 날이 많았어요")
                st.info("💡 예측으로는 큰 문제가 없지만, 작년 이 시기 기록으로는 " + ", ".join(parts) + ". "
                        "그 시각에 이용한다면 근처 대여소도 함께 확인해 두세요.")
            else:
                st.success("💡 오늘 이 시간대엔 큰 문제 없이 이용할 수 있을 것으로 보여요.")

    # ④ 유형 설명 카드 — [원리] 평소 모습(2025년 평일 중앙값)과 오늘 예측을 나란히 보여 준다.
    st.markdown("#### 유형별 설명")
    # [수정 2026-11-17] 4장을 한 줄에 두면 좁은 화면에서 글자가 세로로 길어져, 2장씩 두 줄로 놓는다.
    cols = [c for _ in range(0, len(tmeta["types"]), 2) for c in st.columns(2)]
    for col, t in zip(cols, tmeta["types"]):
        s = type_today(t["key"])
        with col.container(border=True):
            st.markdown(f"<div style='border-top:6px solid {t['color']};padding-top:6px'>"
                        f"<b>{t['emoji']} {t['name']}</b></div>", unsafe_allow_html=True)
            st.caption(t["desc"])
            st.markdown(f"**대여소 {t['n_stations']}곳** · 하루 평균 이용 약 {t['daily_volume']:.0f}건"
                        + (f" · 상시 과밀 {s['chronic']}곳" if s and s["chronic"] else ""))
            st.markdown(f"평소 출근(7~9시): {flow_text(t['am_return_per_rental'])}")
            st.markdown(f"평소 퇴근(18~20시): {flow_text(t['pm_return_per_rental'])}")
            if s:
                st.markdown(f"**오늘 {period_word} 예측:** {net_text(s['net'])}")

    # ⑥ 분석 방법 — [원리] 쉬운 설명 아래에 근거를 작게 남겨 발표·검증 때 확인할 수 있게 한다.
    # [수정 2026-11-17] 한 문단 설명 → 항목별 목록. 실루엣 점수 수준 해석, 유형 고정·카드 숫자(중앙값) 설명을 더했다.
    # [원리] 숫자는 모두 station_types_meta.json(계산 스크립트 결과)에서 읽어, 계산을 다시 하면 설명도 함께 바뀐다.
    with st.expander("ℹ️ 분석 방법 (K-Means)"):
        sil = tmeta["silhouette_by_k"]
        k = str(tmeta["k"])
        best_k = max(sil, key=sil.get)
        sil_text = ", ".join(f"{n}개 {v}" for n, v in sil.items())
        st.markdown(
            f"**무엇으로 나눴나요?**\n"
            f"- {tmeta['period']} 기록으로 대여소마다 4가지 값을 계산했어요.\n"
            f"  - 출근 7~9시에 자전거가 들어오는지 나가는지: (반납 − 대여) ÷ (반납 + 대여)\n"
            f"  - 퇴근 18~20시에 같은 계산\n"
            f"  - 하루 평균 이용 건수(큰 값의 영향을 줄이려고 로그로 바꿈)\n"
            f"  - 하루 이용 중 출퇴근 6시간이 차지하는 비율\n\n"
            f"**어떻게 묶었나요?**\n"
            f"- 단위가 다른 4가지 값을 같은 무게로 비교하도록 맞춘(표준화) 뒤, "
            f"K-Means(비슷한 것끼리 자동으로 묶는 방법)로 {k}개 유형으로 나눴어요.\n"
            f"- 유형 이름은 사람이 하나씩 붙이지 않고, 각 묶음의 평균 모습을 보고 정한 규칙으로 붙였어요.\n\n"
            f"**얼마나 뚜렷하게 나뉘나요?**\n"
            f"- 묶음이 얼마나 뚜렷한지 보는 실루엣 점수(−1~1, 1에 가까울수록 뚜렷)는 {sil_text}였어요.\n"
            f"- {k}개({sil[k]})는 가장 높은 {best_k}개({sil[best_k]})와 거의 같으면서, 집·회사·역 앞처럼 이용자에게 "
            f"의미 있는 구분을 줘서 골랐어요.\n"
            f"- 0.3대는 **대체로 구분되지만, 두 유형 경계에 걸친 대여소도 있는 수준**이에요. "
            f"그래서 일부 대여소는 이웃 유형의 성격도 함께 가질 수 있어요.\n\n"
            f"**알아 두실 점**\n"
            f"- 유형은 2025년 기록으로 한 번 정한 것이라 **고른 날짜와 관계없이 같아요.** "
            f"지도·요약의 '오늘' 값만 사이드바에서 고른 날짜·시간대의 예측이에요.\n"
            f"- 유형 카드의 '평소 출근·퇴근' 숫자(예: 1대 나갈 때 5.9대 들어와요)는 그 유형 대여소들의 **중앙값**이에요.\n"
            f"- {tmeta['note']}"
        )

# ------------------------------------------
# Mode 5: 재배치 도우미 (따릉이 재배치 담당자용)
# ------------------------------------------
# [수정 2026-11-17 메뉴 분리] 2번 메뉴 안 '운영자 보기' 스위치에 있던 화면들을 별도 메뉴로 옮겼다(사용자 결정).
# [왜] 2번은 이용자용(내 대여소가 어떤 곳인지)이고, 이 화면들은 재배치 담당자용이라 대상이 다르다. '운영자'는 서비스 운영자로
#      오해될 수 있어 '따릉이 재배치 담당자'로 부른다. 누구나 볼 수 있다(로그인 없음).
# [역할] (가) 구·유형 현황 (나) 옮길 방향 후보 (다) 흐름 예보 (라) 이동 흐름 화살표 중 하나를 골라 본다.
# [원리] 2번 메뉴와 같은 예측(get_predictions)과 대여소 유형표를 합친 표(d)를 쓰고, 고른 화면 하나만 그린다.
elif mode == MENU5:
    df_predicted, info = get_predictions()
    types, tmeta = load_station_types()
    with st.container(key="subheader_box_5"):
        st.subheader("🚚 재배치 도우미 — 따릉이 재배치 담당자를 위한 화면")
    st.caption("어느 구·유형에서 자전거가 모자라거나 남는지, 어디서 어디로 옮기면 좋을지, 자전거가 어디로 흐르는지 볼 수 있어요. "
               "재배치 담당자를 위한 화면이지만 누구나 볼 수 있어요.")
    if types is None:
        st.error("대여소 유형 파일(data/station_types.csv)이 없습니다. scripts/build_station_types.py로 만들어 주세요.")
        st.stop()
    d = df_predicted.merge(types, left_on="대여소ID", right_on="station_id", how="left")
    op_view = st.segmented_control("재배치 도우미 화면", ["구·유형 현황", "옮길 방향 후보", "흐름 예보", "이동 흐름"],
                                   default="구·유형 현황", key="op_tab", label_visibility="collapsed")
    if op_view == "구·유형 현황":
        st.markdown("**(가) 자치구 × 유형별 오늘 예측**")
        gt = d.assign(부족=d["위험상태"].str.startswith("부족"), 포화=d["위험상태"].str.startswith("포화"),
                      유형=d["type_emoji"].fillna("") + " " + d["type_name"].fillna("미분류"))
        # [수정 2026-11-17 표 이름 정리] 열 이름을 쉬운 말로 바꾸고 단위(곳·대)를 붙이며, 결론을 '추천 방향' 한 칸으로 준다.
        # [원리] 증감은 숫자 그대로 두고(머리글로 정렬 가능) 표시 형식만 +/−와 '대'로 바꾼다(st.column_config).
        #        추천 방향 규칙: 부족 예상이 있으면 🚚 재배치 받기 / 대여소당 +3대 이상 남고 거치대 거의 참이 있으면 📦 수거 후 재배치 / 그 밖 ✅ 유지.
        #        3대는 '눈에 띄게 남는 수준'으로 정한 값(가정)이다. 정렬은 🚚 → 📦 → ✅, 그 안에서 많이 모자란/남는 순.
        gt = (gt.groupby(["클러스터", "유형"])
                .agg(n=("대여소ID", "count"), net=("순수요_NetFlow", "mean"),
                     short=("부족", "sum"), full=("포화", "sum"), chronic=("상시과밀", "sum"))
                .reset_index())
        gt["net"] = gt["net"].round(1)
        gt["추천 방향"] = np.select(
            [gt["short"] > 0, (gt["net"] >= 3) & (gt["full"] > 0)],
            ["🚚 재배치 받기 (보충 필요)", "📦 수거 후 재배치 (여유분 이동)"], default="✅ 유지")
        rank = gt["추천 방향"].str[0].map({"🚚": 0, "📦": 1, "✅": 2})
        gt = gt.assign(_r=rank, _k=np.where(rank == 1, -gt["net"], gt["net"])).sort_values(["_r", "_k"])
        gt = gt[["클러스터", "유형", "n", "net", "추천 방향", "short", "full", "chronic"]].rename(columns={
            "클러스터": "자치구", "n": "대여소 수", "net": "대여소당 증감 (+남음/−모자람)",
            "short": "자전거 부족 예상", "full": "거치대 거의 참", "chronic": "평소에도 넘침"})
        st.dataframe(one_based(gt), width="stretch", height=300, column_config={
            "대여소 수": st.column_config.NumberColumn(format="%d곳"),
            "대여소당 증감 (+남음/−모자람)": st.column_config.NumberColumn(
                format="%+.1f대", help="대여소 한 곳당 평균 (예측 반납 − 예측 대여). +면 자전거가 남고, −면 모자랍니다."),
            "추천 방향": st.column_config.TextColumn(
                help="🚚 부족 예상 대여소가 있음 → 다른 곳에서 가져와 채우기 / 📦 대여소당 3대 이상 남고 거치대가 거의 참 → 남는 자전거를 거둬 옮기기 / ✅ 손댈 필요 없음"),
            "자전거 부족 예상": st.column_config.NumberColumn(format="%d곳", help="끝날 때 남은 자전거가 3대 이하로 예측되거나, 작년 이 시기 절반 이상의 날에 3대 이하였던 대여소 수"),
            "거치대 거의 참": st.column_config.NumberColumn(format="%d곳", help="끝날 때 자전거가 거치대 수 − 3대 이상으로 예측된 대여소 수(반납은 연결 반납으로 가능)"),
            "평소에도 넘침": st.column_config.NumberColumn(format="%d곳", help="2025년 평소(중앙값) 재고가 거치대 수보다 많은 대여소 수"),
        })
        st.caption("열 이름에 마우스를 올리면 설명이 나옵니다. 급한 순서(🚚 → 📦 → ✅)로 정렬했습니다.")
    elif op_view == "옮길 방향 후보":
        st.markdown("**(나) 옮길 방향 후보** — 🚚 재배치 받을 곳 ← 가까운 📦 수거할 곳")
        need = d[d["위험상태"].str.startswith("부족")]
        have = d[d["예측_잔여수량"] >= d["총거치대수"]]
        pairs = []
        for _, n in need.iterrows():
            h = have.assign(거리=haversine_m(n["위도"], n["경도"], have["위도"].to_numpy(float), have["경도"].to_numpy(float)))
            for _, s in h.nsmallest(2, "거리").iterrows():
                # [수정 2026-11-17] 열 이름을 🚚 받을 곳 / 📦 수거할 곳으로 통일하고 숫자에 단위 '대'를 붙인다.
                # [수정 2026-11-18] 받을 곳의 남을 자전거도 숫자 대신 구간(inv_text)으로. 작년 기록으로 부족이 된 곳은 그 사실을 붙인다.
                pairs.append({"🚚 재배치 받을 곳": f"{n['대여소명']} ({n['클러스터']})",
                              "받을 곳: 남을 자전거": inv_text(n["예측_잔여수량"], n["총거치대수"])
                              + (" · 작년 이 시기 자주 부족" if "작년" in str(n["부족근거"]) else ""),
                              "📦 수거할 곳": f"{s['대여소명']} ({s['클러스터']})",
                              "수거할 곳: 남을 자전거": inv_text(s["예측_잔여수량"], s["총거치대수"]),
                              "수거할 곳: 거치대": int(s["총거치대수"]), "거리(직선)": fmt_distance(s["거리"])})
        if pairs:
            st.dataframe(one_based(pd.DataFrame(pairs)), width="stretch", column_config={
                "받을 곳: 남을 자전거": st.column_config.TextColumn(
                    help="시간대가 끝날 때 남을 자전거(추정 구간). 추정 오차가 커서 숫자 대신 구간으로 보여 줍니다."),
                "수거할 곳: 남을 자전거": st.column_config.TextColumn(
                    help="거치대 수를 넘는 예측은 숫자가 과장일 수 있어(실제로는 그 전에 재배치됨) '거치대 초과'로만 표시합니다."),
                "수거할 곳: 거치대": st.column_config.NumberColumn(format="%d석"),
            })
        else:
            st.caption("이 시간대에는 부족 위험 대여소가 없거나, 자전거가 남는 대여소가 없습니다.")
        st.caption("방향 제안일 뿐이며 옮길 대수·차량 경로는 계산하지 않습니다. 📦 수거할 곳 = 끝날 때 자전거가 거치대 수 이상 남을 곳. "
                   "직선거리 기준이라 실제 이동 거리는 더 깁니다. 분석 대상 60곳 안에서만 짝을 찾습니다.")
    elif op_view == "흐름 예보":
        render_flow_forecast(df_predicted, info)
    elif op_view == "이동 흐름":   # [추가 2026-11-17] 목표 3: 대여 후 반납 위치 이동 화살표
        render_od_flow(df_predicted, info)

# ------------------------------------------
# Mode 3: 시간대별 대여·반납 예측 (선 그래프)
# ------------------------------------------
elif mode == "3. 시간대별 대여·반납 예측 그래프":
    # [역할] 하루치 시간별 예측 가져오기. [원리] C안이면 날짜 예측 결과(24시간), 아니면 기존 참조일 재생.
    if use_lt:
        require_lt()
        daily, sims = run_daily_lt(selected_date, weather_items, lt_model)
        show_lt_banner(sims)
    else:
        ref_date = get_reference_date()
        daily = run_daily_prediction(model_type, ref_date)
        if daily is None:
            st.warning(f"참조일({ref_date})의 데이터가 없습니다. 다른 날짜를 선택해 주세요.")
            st.stop()

    weekday = WEEKDAYS[selected_date.weekday()]
    with st.container(key="subheader_box_3"):
        st.subheader(f"📈 {day_text(selected_date)} 시간대별 예측 대여·반납 대수 · {model_title}")

    picked_id = selected_station_id
    if picked_id is not None and not (daily["대여소ID"] == picked_id).any():
        st.caption("선택한 대여소는 이 날짜 데이터가 없어 전체 대여소 합계를 보여드립니다.")
        picked_id = None
    hourly = hourly_totals(daily, picked_id)

    if picked_id is None:
        target_name = "전체 대여소 합계"
    else:
        target_name = daily.loc[daily["대여소ID"] == picked_id, "대여소명"].iloc[0]
    st.markdown(f"##### {target_name}")
    st.line_chart(
        hourly,
        color=["#e63946", "#1d7ae0"],  # 예측 대여 = 빨강, 예측 반납 = 파랑
        x_label="시간 (시)",
        y_label="자전거 대수",
        height=420,
    )
    with st.expander("시간별 예측값 표 보기"):
        # [수정 2026-11-17] 숫자에 단위(대)를 붙여 보여 준다(값은 그대로)
        st.dataframe(hourly.round(1), width="stretch", column_config={
            "예측 대여": st.column_config.NumberColumn(format="%.1f대"),
            "예측 반납": st.column_config.NumberColumn(format="%.1f대")})

# ------------------------------------------
# Mode 4: 따릉이 길찾기 (출발지 근처 대여 → 도착지 근처 반납)
# ------------------------------------------
elif mode == "4. 가까운 대여소 길찾기":
    if not find_data_file(STATIONS_FILE).exists():
        st.error(f"{STATIONS_FILE} 파일을 찾을 수 없습니다. data/ 폴더에 넣어 주세요.")
        st.stop()

    stations = load_station_locations()
    with st.container(key="subheader_box_4"):
        st.subheader("🧭 출발지 근처에서 빌려 도착지 근처에 반납하기")
    st.caption(
        "출발지·도착지를 장소 이름이나 주소로 검색하면 각각 가장 가까운 대여소를 찾고, "
        "대여소 사이 **자전거 탑승 예상 시간**을 보여 줍니다. 대상 대여소는 분석용 60곳입니다."
    )
    if not kakao_key():
        st.error(
            "카카오 API 키(KAKAO_REST_KEY)가 없습니다. 앱을 실행하기 전에 환경변수 `KAKAO_REST_KEY`를 설정하거나, "
            '`.streamlit/secrets.toml`에 `KAKAO_REST_KEY = "발급받은 REST API 키"`를 넣고 앱을 다시 실행해 주세요.'
        )
        st.stop()

    # 버튼을 눌렀을 때만 검색 (입력할 때마다 API를 부르지 않도록)
    with st.form("route_form"):
        f1, f2 = st.columns(2)
        q_origin = f1.text_input("출발지", placeholder="예: 서울역, 서울 중구 세종대로 110")
        q_dest = f2.text_input("도착지", placeholder="예: 광화문")
        submitted = st.form_submit_button("🔍 대여소 찾기", type="primary")
    if submitted:
        st.session_state["route_query"] = (q_origin.strip(), q_dest.strip())
    q_origin, q_dest = st.session_state.get("route_query", ("", ""))
    if not q_origin or not q_dest:
        st.info("출발지와 도착지를 모두 입력하고 '대여소 찾기'를 눌러 주세요.")
        st.stop()


    p1, p2 = st.columns(2)
    origin = pick_place("출발지", q_origin, p1)
    dest = pick_place("도착지", q_dest, p2)
    if origin is None or dest is None:
        st.stop()

    forecast, forecast_note = station_forecast()

    def with_status(cands):
        """후보 표에 예측 재고 상태를 붙인다(예측이 없으면 '정보 없음')."""
        cands = cands.copy()
        if forecast is None:
            cands["예측상태"] = "정보 없음"
            cands["예측잔여"] = "-"
        else:
            cands["예측상태"] = cands["station_id"].map(forecast["위험상태"]).fillna("정보 없음")
            # [수정 2026-11-17] 거치대를 넘는 예측은 숫자 대신 '거치대 초과'로 표시(inv_text 참고)
            cands["예측잔여"] = [inv_text(forecast["예측_잔여수량"].get(sid), forecast["총거치대수"].get(sid))
                                 if sid in forecast.index else "-" for sid in cands["station_id"]]
        return cands

    rent_c = with_status(nearest_to_point(stations, origin["lat"], origin["lon"]))
    ret_c = with_status(nearest_to_point(stations, dest["lat"], dest["lon"]))
    for label, place, cands in (("출발지", origin, rent_c), ("도착지", dest, ret_c)):
        if cands.empty:
            st.error(
                f"{label} '{place['name']}'에서 직선 {fmt_distance(MAX_STATION_DIST_M)} 안에 대상 대여소가 없습니다. "
                "서울 시내의 다른 장소로 검색해 주세요."
            )
    if rent_c.empty or ret_c.empty:
        st.stop()

    # [수정 2026-11-17] max_extra_m: 피하는 대신 더 가도 되는 최대 거리. [왜] '포화'는 거치대가 거의 찬 것일 뿐 반납은 된다
    #        (다른 자전거에 연결). 포화가 흔하므로 무조건 피하면 훨씬 먼 곳으로 안내된다 → 300m 이내로 더 가는 경우만 피한다.
    def choose(cands, bad_word, role, max_extra_m=None):
        """가까운 순서로, 예측상 bad_word(부족/포화)가 아닌 첫 대여소를 고른다."""
        ok = cands[~cands["예측상태"].str.contains(bad_word)]
        if max_extra_m is not None and not ok.empty and \
                ok.iloc[0]["straight_m"] - cands.iloc[0]["straight_m"] > max_extra_m:
            st.caption(f"💡 {role}: 가장 가까운 '{cands.iloc[0]['station_name']}'은(는) 거치대가 거의 찰 것으로 예측되지만, "
                       "반납은 가능하고(다른 자전거에 연결) 대안이 멀어 그대로 안내합니다.")
            return cands.iloc[0]
        if ok.empty:
            st.warning(f"{role} 후보 {len(cands)}곳이 모두 '{bad_word} 위험'으로 예측됩니다. 가장 가까운 곳을 표시합니다.")
            return cands.iloc[0]
        pick = ok.iloc[0]
        if pick["station_id"] != cands.iloc[0]["station_id"]:
            st.caption(
                f"💡 {role}: 가장 가까운 '{cands.iloc[0]['station_name']}'은(는) '{bad_word} 위험'으로 예측되어 "
                f"다음으로 가까운 '{pick['station_name']}'을(를) 골랐습니다."
            )
        return pick

    rent = choose(rent_c, "부족", "대여")
    ret = choose(ret_c, "포화", "반납", max_extra_m=300)
    o = (origin["lat"], origin["lon"])
    d = (dest["lat"], dest["lon"])
    s1 = (float(rent["lat"]), float(rent["lon"]))
    s2 = (float(ret["lat"]), float(ret["lon"]))

    same_station = rent["station_id"] == ret["station_id"]
    route = None if same_station else bike_leg(s1, s2)

    st.markdown(
        f"**🚲 대여:** {rent['station_name']} ({rent['gu']}) → **반납:** {ret['station_name']} ({ret['gu']})"
    )
    if same_station:
        st.info("출발지와 도착지에서 가장 가까운 대여소가 같습니다. 자전거를 탈 필요 없이 걸어가는 편이 낫습니다.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("자전거 탑승 예상 시간", f"약 {max(1, round(route['duration_s'] / 60))}분",
                  help="대여소 → 대여소 구간만의 시간입니다. 걷는 시간과 대여·반납 절차 시간은 포함하지 않습니다.")
        m2.metric("자전거 이동거리", fmt_distance(route["distance_m"]))
        m3.metric("출발지 → 대여소 (직선)", fmt_distance(rent["straight_m"]))
        m4.metric("반납 대여소 → 도착지 (직선)", fmt_distance(ret["straight_m"]))
        if route["estimated"]:
            st.warning(
                "자전거 경로 서버에 연결하지 못해 직선거리 × 1.3, 평균 시속 15km로 추정했습니다(점선). "
                "잠시 후 다시 시도하면 실제 경로로 계산합니다."
            )
        if rent["straight_m"] + ret["straight_m"] > route["distance_m"]:
            st.info("대여소까지 걷는 거리가 자전거 이동거리보다 깁니다. 걸어가는 편이 나을 수 있습니다.")
    for role, row in (("대여", rent), ("반납", ret)):
        if row["straight_m"] > WARN_STATION_DIST_M:
            st.warning(f"{role} 대여소까지 직선으로 {fmt_distance(row['straight_m'])} 떨어져 있습니다. 걸어서 시간이 더 걸릴 수 있습니다.")
    st.caption(f"ℹ️ {forecast_note}")

    with st.expander("가까운 대여소 후보 보기 (직선거리 순)"):
        t1, t2 = st.columns(2)
        for col, title, cands in ((t1, "출발지 근처 (대여)", rent_c), (t2, "도착지 근처 (반납)", ret_c)):
            col.markdown(f"**{title}**")
            view = pd.DataFrame({
                "대여소": cands["station_name"],
                "구": cands["gu"],
                "직선거리": cands["straight_m"].map(fmt_distance),
                "예측 상태": cands["예측상태"],
                "예측 잔여": cands["예측잔여"],
            })
            col.dataframe(one_based(view), width="stretch")

    # 지도: 자전거 경로(빨강), 걷는 구간은 직선 점선(회색, 참고용)
    path = [list(s1), list(s2)] if route is None else route["path"]
    pts = [list(o), list(d), list(s1), list(s2), *path]
    # fit_bounds는 화면 맞춤 애니메이션이 중간에 끊겨 엉뚱한 위치로 열리는 경우가 있어, 중심·확대 단계를 직접 계산한다
    center, zoom = fit_view(pts)
    m5 = folium.Map(location=center, zoom_start=zoom)
    for _, r in stations[~stations["station_id"].isin([rent["station_id"], ret["station_id"]])].iterrows():
        folium.CircleMarker(location=[r["lat"], r["lon"]], radius=4, color="#8d99ae", fill=True,
                            fill_opacity=0.8, tooltip=f"{r['station_name']} ({r['gu']})").add_to(m5)
    for a, b in ((o, s1), (s2, d)):
        folium.PolyLine([list(a), list(b)], color="#8d99ae", weight=3, dash_array="4 6",
                        tooltip="걷는 구간 (직선, 참고용)").add_to(m5)
    if route is not None:
        style = {"color": "#e63946", "weight": 5, "opacity": 0.9}
        if route["estimated"]:
            style["dash_array"] = "8"
        folium.PolyLine(path, tooltip=f"자전거 약 {fmt_distance(route['distance_m'])}", **style).add_to(m5)
    for loc, tip, color, icon in (
        (o, f"출발지: {origin['name']}", "green", "home"),
        (s1, f"대여: {rent['station_name']}", "red", "star"),
        (s2, f"반납: {ret['station_name']}", "blue", "flag"),
        (d, f"도착지: {dest['name']}", "black", "map-marker"),
    ):
        folium.Marker(location=list(loc), tooltip=tip,
                      icon=folium.Icon(color=color, icon=icon, prefix="glyphicon")).add_to(m5)
    st_folium(m5, width="100%", height=480, returned_objects=[], key="map5_route")
    st.caption(
        "주소 검색: Kakao 로컬 API · 경로: © OpenStreetMap contributors (ODbL) · OSRM, routing.openstreetmap.de (FOSSGIS). "
        "공개 데모 서버를 쓰므로 비상업·소규모 용도로만 사용하세요."
    )