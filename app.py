import datetime as dt
import time
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


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
MAX_DAYS_AHEAD = 365  # 날짜 선택 가능 범위: 오늘 ~ 1년 뒤

# 자전거 경로: FOSSGIS가 운영하는 공개 OSRM 서버(자전거 프로필). 비상업·소규모 사용용이며 초당 1회 이하로 호출해야 한다.
ROUTING_BASE = "https://routing.openstreetmap.de/routed-bike"
ROUTING_USER_AGENT = "ddareungi-esg-demo/1.0"  # 서버 이용 규칙상 앱을 식별하는 값 (연락처를 덧붙이는 것을 권장)
ROUTING_TIMEOUT_S = 10
NEAREST_CANDIDATES = 3  # 직선거리 기준 가까운 후보 수 (이 중 자전거 경로가 가장 짧은 곳을 고름)


# ==========================================
# 1. 페이지 및 기본 레이아웃 설정
# ==========================================
st.set_page_config(
    page_title="따릉이 출퇴근 수요 예측",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 AI 기반 따릉이 출퇴근 수요 예측 및 대여소 안내 서비스")
st.caption(
    "출퇴근 피크 시간대 주거지 ↔ 업무지 간 역방향 수요 예측, K-Means 클러스터 분석, "
    "시간대별 예측 그래프, 자전거 경로 기반 가까운 대여소 안내를 제공합니다."
)

# 사이드바: 접기(◀) 버튼을 '통합 운영 제어판' 제목과 같은 줄로 올리고, 내용을 위로 당긴다.
# (스트림릿 내부 구조(data-testid)에 의존하므로, 버전이 바뀌면 어긋날 수 있다)
st.markdown(
    """
    <style>
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
    }
    .stat-row {
        display: grid;
        grid-template-columns: repeat(3, 6.5rem);
        column-gap: 0.25rem;
        justify-content: end;
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
    return stations


@st.cache_data(show_spinner=False)
def load_stations():
    return read_stations_csv()[["station_id", "station_name", "gu", "group"]]


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


def sidebar_station_picker(disabled=False):
    """사이드바 대여소 드롭다운. 선택한 station_id를 돌려주고, '전체 대여소'면 None."""
    if not find_data_file(STATIONS_FILE).exists():
        return None
    stations = load_stations()
    names = stations["station_name"].fillna(stations["station_id"])
    duplicated = names.duplicated(keep=False)  # 이름이 같은 대여소는 ID를 덧붙여 구분
    labels = np.where(duplicated, names + " (" + stations["station_id"] + ")", names)
    lookup = dict(zip(labels, stations["station_id"]))
    choice = st.sidebar.selectbox(
        "📍 대여소 선택",
        ["전체 대여소"] + sorted(lookup),
        disabled=disabled,
        help="지도(1·2번 메뉴)는 선택한 대여소로 확대되고, 선 그래프(3번 메뉴)는 해당 대여소만 보여주며, "
             "4번 메뉴는 이 대여소에서 가장 가까운 대여소까지의 경로를 그립니다.",
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

    out["예측_잔여수량"] = (
        (out["현재자전거수"] + out["순수요_NetFlow"])
        .round()
        .clip(lower=0, upper=out["총거치대수"])
        .astype(int)
    )

    conditions = [
        (out["예측_잔여수량"] <= 3),
        (out["예측_잔여수량"] >= out["총거치대수"] - 3),
    ]
    choices = ["부족 위험 🔴 (대여 불가)", "포화 위험 🔵 (반납 불가)"]
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

    # 관측 기상 (모델 입력으로 쓰인 1시간 전 관측값; 서울 단일 관측소라 모든 대여소가 같음)
    wx = sub.groupby("timestamp").agg(
        temp=("temp_c_lag1h", "mean"), precip=("precip_mm_lag1h", "mean")
    )
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


def haversine_m(lat1, lon1, lat2, lon2):
    """두 지점 사이의 직선거리(m). lat2/lon2는 배열이어도 된다."""
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    d_phi = p2 - p1
    d_lambda = np.radians(lon2) - np.radians(lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(d_lambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_stations(stations, station_id, k=NEAREST_CANDIDATES):
    """선택한 대여소와 직선거리 기준으로 가까운 후보 k곳. (선택 대여소, 후보 표) 또는 None."""
    me_rows = stations[stations["station_id"] == station_id]
    others = stations[stations["station_id"] != station_id].copy()
    if me_rows.empty or others.empty:
        return None
    me = me_rows.iloc[0]
    others["straight_m"] = haversine_m(
        me["lat"], me["lon"], others["lat"].to_numpy(dtype=float), others["lon"].to_numpy(dtype=float)
    )
    return me, others.nsmallest(k, "straight_m").reset_index(drop=True)


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
def find_bike_route(me_lat, me_lon, candidates):
    """후보(직선거리 가까운 순) 중 자전거 경로가 가장 짧은 대여소를 고르고 그 경로를 돌려준다.
    candidates: ((lat, lon), ...). 경로 호출이 실패하면 예외를 던진다(실패 결과가 캐시되지 않도록)."""
    best = 0
    if len(candidates) > 1:
        try:
            coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in [(me_lat, me_lon), *candidates])
            destinations = ";".join(str(i) for i in range(1, len(candidates) + 1))
            data = osrm_get(
                f"table/v1/driving/{coords}",
                {"sources": "0", "destinations": destinations, "annotations": "distance"},
            )
            reachable = [(d, i) for i, d in enumerate(data["distances"][0]) if d is not None]
            if reachable:
                best = min(reachable)[1]
        except Exception:
            best = 0  # 거리 비교에 실패하면 직선거리 기준 1순위 후보로 진행
        finally:
            time.sleep(1.1)  # 공개 서버 이용 규칙: 초당 1회 이하

    lat, lon = candidates[best]
    data = osrm_get(
        f"route/v1/driving/{me_lon:.6f},{me_lat:.6f};{lon:.6f},{lat:.6f}",
        {"overview": "full", "geometries": "geojson"},
    )
    route = data["routes"][0]
    return {
        "index": best,
        "distance_m": float(route["distance"]),
        "duration_s": float(route["duration"]),
        "path": [[la, lo] for lo, la in route["geometry"]["coordinates"]],  # GeoJSON은 (경도, 위도) 순서
    }


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
st.sidebar.header("🔍 통합 운영 제어판")
mode = st.sidebar.radio(
    "메뉴 선택",
    [
        "1. 실시간 수급 불균형 지도",
        "2. K-Means 클러스터 분석",
        "3. 시간대별 대여·반납 예측 (선 그래프)",
        "4. 가까운 대여소 찾기 (경로 지도)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 예측 AI 모델 및 조건 설정")
model_type = st.sidebar.selectbox("예측 모델 선택", MODEL_OPTIONS)

today = pd.Timestamp.now(tz="Asia/Seoul").date()
selected_date = st.sidebar.date_input(
    "날짜 선택 (오늘 이후)",
    value=today,
    min_value=today,
    max_value=today + dt.timedelta(days=MAX_DAYS_AHEAD),
)
period_label = st.sidebar.selectbox("피크 시간대 선택", list(PERIOD_OPTIONS))
selected_station_id = sidebar_station_picker()
st.sidebar.caption(
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


# ==========================================
# 5. 메인 화면 분기
# ==========================================

# ------------------------------------------
# Mode 1: 실시간 지도 시각화
# ------------------------------------------
if mode == "1. 실시간 수급 불균형 지도":
    df_predicted, info = get_predictions()
    window_text = f"{selected_date} {period_label} ({info['hour_min']:02d}~{info['hour_max']:02d}시)"
    st.subheader(f"📊 [{model_type} 모델] {window_text} 기준 대여소별 Net-flow(반납-대여) 예측 현황")

    shortage_cnt = len(df_predicted[df_predicted["위험상태"].str.contains("부족")])
    overflow_cnt = len(df_predicted[df_predicted["위험상태"].str.contains("포화")])

    st.markdown("---")
    left_col, right_col = st.columns([2, 1])

    with left_col:
        title_col, stats_col = st.columns([1, 1.2], vertical_alignment="center")
        title_col.markdown(
            f'<div class="compact-heading">🗺️ 대여소별 위험도 지도 ({model_type})</div>',
            unsafe_allow_html=True,
        )
        # 값과 변화 태그를 한 줄에 놓기 위해 st.metric 대신 직접 만든 HTML을 사용 (줄바꿈 없이 한 줄로 조립)
        stats_html = (
            '<div class="stat-row">'
            '<div class="stat"><div class="stat-label">분석 대여소</div>'
            f'<div class="stat-line"><span class="stat-value">{len(df_predicted)}개소</span></div></div>'
            '<div class="stat"><div class="stat-label">부족 위험 '
            '<span class="stat-help" title="예측 잔여 자전거가 3대 이하라 대여가 어려운 대여소">?</span></div>'
            f'<div class="stat-line"><span class="stat-value">{shortage_cnt}곳</span>'
            '<span class="stat-pill down">↓ -경고</span></div></div>'
            '<div class="stat"><div class="stat-label">포화 위험 '
            '<span class="stat-help" title="예측 잔여 자전거가 거치대 여유 3칸 이하라 반납이 어려운 대여소">?</span></div>'
            f'<div class="stat-line"><span class="stat-value">{overflow_cnt}곳</span>'
            '<span class="stat-pill up">↑ +주의</span></div></div>'
            '</div>'
        )
        stats_col.markdown(stats_html, unsafe_allow_html=True)
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
                f"• 예측 순수요(Net-flow): {row['순수요_NetFlow']:+.1f}대<br>"
                f"• <b>예측 잔여: {row['예측_잔여수량']}대</b> / (총 {int(row['총거치대수'])}석)"
                f"</div>"
            )

            folium.Marker(
                location=[row["위도"], row["경도"]],
                popup=folium.Popup(popup_text, max_width=300, show=is_selected),
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
            '<div class="compact-heading">📋 대여소별 Net-flow 상세표</div>', unsafe_allow_html=True
        )
        table_cols = ["대여소명", "순수요_NetFlow", "위험상태"]
        if info["has_actual"]:
            table_cols.insert(2, "실제_NetFlow")
        table = df_predicted[table_cols].copy()
        for c in ("순수요_NetFlow", "실제_NetFlow"):
            if c in table.columns:
                table[c] = table[c].round(1)
        table = table.rename(columns={"실제_NetFlow": "참조일_실제_NetFlow"})
        st.dataframe(one_based(table), use_container_width=True, height=400)
        if info["has_actual"]:
            st.caption(
                f"참조일_실제_NetFlow: 참조일({info['ref_date']}) 같은 시간대의 실제 (반납 - 대여) 합계로, "
                "모델이 그날을 얼마나 맞혔는지 비교하는 용도입니다."
            )

# ------------------------------------------
# Mode 2: K-Means 클러스터 분석
# ------------------------------------------
elif mode == "2. K-Means 클러스터 분석":
    df_predicted, info = get_predictions()
    st.subheader("🏙️ 군집 분석(K-Means) 기반 업무 밀집가 수급 불균형 클러스터")
    st.markdown("수급 불균형이 심각하게 연동되는 주요 거점 클러스터(예: 여의도, 마곡, 강남 등 업무 밀집가)를 그룹화하여 분석합니다.")

    cluster_group = (
        df_predicted.groupby("클러스터")
        .agg(
            대여소수=("대여소ID", "count"),
            평균순수요=("순수요_NetFlow", "mean"),
            평균잔여수량=("예측_잔여수량", "mean"),
        )
        .round(1)
        .reset_index()
    )

    st.dataframe(one_based(cluster_group), use_container_width=True)

    selected_station = station_row(df_predicted, selected_station_id)
    m_cluster = make_base_map(df_predicted, selected_station)
    for _, row in df_predicted.iterrows():
        is_selected = (
            selected_station is not None and row["대여소ID"] == selected_station["대여소ID"]
        )
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=10,
            popup=folium.Popup(
                f"<b>{row['클러스터']}</b><br>{row['대여소명']}", max_width=300, show=is_selected
            ),
            color="orange",
            fill=True,
            fill_color="orange",
            fill_opacity=0.7,
        ).add_to(m_cluster)

    st_folium(
        m_cluster,
        width="100%",
        height=400,
        returned_objects=[],
        key=map_key("map2", selected_station),
    )

# ------------------------------------------
# Mode 3: 시간대별 대여·반납 예측 (선 그래프)
# ------------------------------------------
elif mode == "3. 시간대별 대여·반납 예측 (선 그래프)":
    ref_date = get_reference_date()
    daily = run_daily_prediction(model_type, ref_date)
    if daily is None:
        st.warning(f"참조일({ref_date})의 데이터가 없습니다. 다른 날짜를 선택해 주세요.")
        st.stop()

    weekday = WEEKDAYS[selected_date.weekday()]
    st.subheader(f"📈 [{model_type} 모델] {selected_date}({weekday}) 시간대별 예측 대여·반납 대수")

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
        st.dataframe(hourly.round(1), use_container_width=True)

# ------------------------------------------
# Mode 4: 가까운 대여소 찾기 (경로 지도)
# ------------------------------------------
elif mode == "4. 가까운 대여소 찾기 (경로 지도)":
    if not find_data_file(STATIONS_FILE).exists():
        st.error(f"{STATIONS_FILE} 파일을 찾을 수 없습니다. data/ 폴더에 넣어 주세요.")
        st.stop()

    stations = load_station_locations()
    st.subheader("🧭 선택한 대여소에서 자전거로 가장 가까운 대여소 찾기")

    def add_other_stations(fmap, skip_ids=()):
        """배경으로 나머지 대여소를 작은 회색 점으로 표시."""
        for _, r in stations[~stations["station_id"].isin(skip_ids)].iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=4,
                color="#8d99ae",
                fill=True,
                fill_opacity=0.8,
                tooltip=f"{r['station_name']} ({r['gu']})",
            ).add_to(fmap)

    found = None if selected_station_id is None else nearest_stations(stations, selected_station_id)

    if found is None:
        st.info("사이드바의 '📍 대여소 선택'에서 대여소를 고르면, 자전거로 가장 가까운 대여소까지의 경로가 지도에 표시됩니다.")
        m5 = folium.Map(location=[stations["lat"].mean(), stations["lon"].mean()], zoom_start=12)
        add_other_stations(m5)
        map5_key = "map5_all"
        route = None
    else:
        me, cands = found
        try:
            route = find_bike_route(
                float(me["lat"]),
                float(me["lon"]),
                tuple((float(r["lat"]), float(r["lon"])) for _, r in cands.iterrows()),
            )
        except Exception:
            route = None  # 경로 서비스 연결 실패 → 아래에서 직선으로 대체

        if route is not None:
            near = cands.iloc[route["index"]]
            path = route["path"]
            minutes = max(1, round(route["duration_s"] / 60))
            route_label = f"자전거 경로 약 {fmt_distance(route['distance_m'])}"
            st.markdown(
                f"**선택:** {me['station_name']} ({me['gu']}) → "
                f"**자전거로 가장 가까운 대여소:** {near['station_name']} ({near['gu']}) · "
                f"{route_label}, 약 **{minutes}분**"
            )
            if route["index"] != 0:
                st.caption(
                    f"💡 직선거리 기준 최단: {cands.iloc[0]['station_name']} · "
                    f"자전거 경로 기준 최단: {near['station_name']} (길을 돌아가야 해서 순서가 달라졌습니다)"
                )
            line_style = {"color": "#e63946", "weight": 5, "opacity": 0.9}
        else:
            near = cands.iloc[0]
            path = [[me["lat"], me["lon"]], [near["lat"], near["lon"]]]
            route_label = f"직선거리 약 {fmt_distance(near['straight_m'])}"
            st.warning("자전거 경로 서비스에 연결하지 못해 직선으로 대신 표시했습니다. 잠시 후 다시 시도해 주세요.")
            st.markdown(
                f"**선택:** {me['station_name']} ({me['gu']}) → "
                f"**가장 가까운 대여소(직선거리):** {near['station_name']} ({near['gu']}) · {route_label}"
            )
            line_style = {"color": "#8d99ae", "weight": 4, "dash_array": "8"}

        lats = [pt[0] for pt in path]
        lons = [pt[1] for pt in path]
        m5 = folium.Map(location=[(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2], zoom_start=15)
        m5.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(60, 60))
        add_other_stations(m5, skip_ids=(me["station_id"], near["station_id"]))

        folium.PolyLine(locations=path, tooltip=route_label, **line_style).add_to(m5)
        folium.Marker(
            location=[me["lat"], me["lon"]],
            tooltip=f"선택: {me['station_name']}",
            popup=folium.Popup(f"<b>{me['station_name']}</b><br>{me['gu']} (선택한 대여소)", max_width=300),
            icon=folium.Icon(color="red", icon="star", prefix="glyphicon"),
        ).add_to(m5)
        folium.Marker(
            location=[near["lat"], near["lon"]],
            tooltip=f"가장 가까운 대여소: {near['station_name']}",
            popup=folium.Popup(f"<b>{near['station_name']}</b><br>{near['gu']}<br>{route_label}", max_width=300),
            icon=folium.Icon(color="blue", icon="flag", prefix="glyphicon"),
        ).add_to(m5)
        map5_key = f"map5_{selected_station_id}"

    st_folium(m5, width="100%", height=480, returned_objects=[], key=map5_key)
    if route is not None:
        st.caption(
            "경로: © OpenStreetMap contributors (ODbL) · OSRM, routing.openstreetmap.de (FOSSGIS). "
            "공개 데모 서버를 쓰므로 비상업·소규모 용도로만 사용하세요."
        )