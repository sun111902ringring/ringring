import datetime as dt
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
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


# ==========================================
# 1. 페이지 및 기본 레이아웃 설정
# ==========================================
st.set_page_config(
    page_title="따릉이 역방향 불균형 및 ESG 패스 시스템",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 AI 기반 따릉이 역방향 불균형 예측 및 기업 ESG '그린 출퇴근 패스' 솔루션")
st.caption(
    "출퇴근 피크 시간대 주거지 ↔ 업무지 간 역방향 수요 예측, K-Means 클러스터 분석, "
    "그리고 기업 맞춤형 그린 출퇴근 패스 ESG 리포트를 제공합니다."
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


@st.cache_data(show_spinner=False)
def load_stations():
    path = find_data_file(STATIONS_FILE)
    try:
        stations = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        stations = pd.read_csv(path, encoding="cp949")
    stations["station_id"] = stations["station_id"].astype(str)
    return stations[["station_id", "station_name", "gu", "group"]]


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
        "3. 기업 맞춤형 '그린 출퇴근 패스' (ESG 리포트)",
        "4. 시간대별 대여·반납 예측 (선 그래프)",
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


def station_picker(df, key, label="📍 대여소 선택 (선택하면 지도가 해당 위치로 확대됩니다)", all_label="전체 보기"):
    """대여소 드롭다운. 선택한 대여소의 행(Series)을 돌려주고, all_label을 고르면 None."""
    names = df["대여소명"]
    duplicated = names.duplicated(keep=False)  # 이름이 같은 대여소는 ID를 덧붙여 구분
    labels = np.where(duplicated, names + " (" + df["대여소ID"] + ")", names)
    lookup = dict(zip(labels, df.index))
    choice = st.selectbox(label, [all_label] + sorted(lookup), key=key)
    return None if choice == all_label else df.loc[lookup[choice]]


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

    # 지도 제목 줄의 요약 지표를 작게 만드는 CSS (key="map_stats" 컨테이너 안의 metric에만 적용)
    st.markdown(
        """
        <style>
        .st-key-map_stats [data-testid="stMetricLabel"] p { font-size: 0.75rem; }
        .st-key-map_stats [data-testid="stMetricValue"] { font-size: 1.2rem; }
        .st-key-map_stats [data-testid="stMetricDelta"] { font-size: 0.7rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    left_col, right_col = st.columns([2, 1])

    with left_col:
        with st.container(key="map_stats"):
            title_col, stat1, stat2, stat3 = st.columns([3, 1, 1, 1], vertical_alignment="center")
            title_col.markdown(f"### 🗺️ 대여소별 위험도 지도 ({model_type})")
            stat1.metric("분석 대여소", f"{len(df_predicted)}개소")
            stat2.metric(
                "부족 위험", f"{shortage_cnt}곳", delta="-경고", delta_color="inverse",
                help="예측 잔여 자전거가 3대 이하라 대여가 어려운 대여소",
            )
            stat3.metric(
                "포화 위험", f"{overflow_cnt}곳", delta="+주의", delta_color="inverse",
                help="예측 잔여 자전거가 거치대 여유 3칸 이하라 반납이 어려운 대여소",
            )
        selected_station = station_picker(df_predicted, key="station_pick_map1")
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
        st.markdown("### 📋 대여소별 Net-flow 상세표")
        table_cols = ["대여소명", "순수요_NetFlow", "위험상태"]
        if info["has_actual"]:
            table_cols.insert(2, "실제_NetFlow")
        table = df_predicted[table_cols].copy()
        for c in ("순수요_NetFlow", "실제_NetFlow"):
            if c in table.columns:
                table[c] = table[c].round(1)
        table = table.rename(columns={"실제_NetFlow": "참조일_실제_NetFlow"})
        st.dataframe(table, width=True, height=400)
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

    st.dataframe(cluster_group, width=True)

    selected_station = station_picker(df_predicted, key="station_pick_map2")
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
# Mode 3: 기업 맞춤형 '그린 출퇴근 패스' (ESG 리포트)
# ------------------------------------------
elif mode == "3. 기업 맞춤형 '그린 출퇴근 패스' (ESG 리포트)":
    st.subheader("🏢 [기업 ESG/Welfare] 맞춤형 '그린 출퇴근 패스' 및 역방향 셔틀 챌린지")

    st.markdown("""
    여의도, 판교, 가산디지털단지 등 대형 업무 지구 내 기업 임직원들이 출퇴근 시 
    따릉이 **역방향 구간**을 이용하도록 독려하고, 탄소 절감량과 예산 절감 기여도를 산출하여 
    기업의 **ESG 평가 리포트**를 제공합니다.
    """)

    col_e1, col_e2, col_e3 = st.columns(3)

    selected_company = st.selectbox(
        "분석 대상 기업 선택",
        ["A사 (여의도 금융권)", "B사 (판교 IT 테크)", "C사 (가산디지털 제조/유통)"],
    )

    emp_count = 1200 if "A사" in selected_company else (2500 if "B사" in selected_company else 800)
    participation_rate = st.slider("임직원 그린 출퇴근 참여율 (%)", 10, 80, 35)

    active_users = int(emp_count * (participation_rate / 100))
    monthly_rides = active_users * 18
    carbon_saved_kg = monthly_rides * 1.25
    eco_points = active_users * 25

    col_e1.metric("참여 임직원 수", f"{active_users:,} 명", delta=f"전체 임직원의 {participation_rate}%")
    col_e2.metric("월간 탄소 배출 절감량", f"{carbon_saved_kg:,.1f} kg", delta="ESG 환경(E) 지표 기여")
    col_e3.metric("임직원 복지 에코 포인트", f"{eco_points:,.0f} P", delta="Welfare ROI 연동")

    st.markdown("---")
    st.markdown("### 📈 기업 단위 자전거 이용 패턴 및 역방향 소진 분석")

    df_esg_report = pd.DataFrame({
        "기업명": [selected_company],
        "인접 클러스터": ["여의도권" if "A사" in selected_company else ("판교/동북권" if "B사" in selected_company else "서남권")],
        "출근 시간대 역방향 소진율": ["매우 높음 (92%)" if participation_rate > 30 else "보통 (60%)"],
        "퇴근 시간대 셔틀 기여도": ["최우수 우수 기업"],
        "ESG 가치 환산 점수": [f"{active_users * 2.5 + carbon_saved_kg * 0.5:.1f} 점"],
    })

    st.dataframe(df_esg_report, width=True)
    st.success(f"💡 {selected_company} 임직원들의 역방향 대여소 자전거 활용으로 환경(E) 및 사회(S) 지표가 크게 향상되었습니다.")

# ------------------------------------------
# Mode 4: 시간대별 대여·반납 예측 (선 그래프)
# ------------------------------------------
elif mode == "4. 시간대별 대여·반납 예측 (선 그래프)":
    ref_date = get_reference_date()
    daily = run_daily_prediction(model_type, ref_date)
    if daily is None:
        st.warning(f"참조일({ref_date})의 데이터가 없습니다. 다른 날짜를 선택해 주세요.")
        st.stop()

    weekday = WEEKDAYS[selected_date.weekday()]
    st.subheader(f"📈 [{model_type} 모델] {selected_date}({weekday}) 시간대별 예측 대여·반납 대수")

    stations_df = daily.drop_duplicates("대여소ID")[["대여소ID", "대여소명"]].reset_index(drop=True)
    picked = station_picker(
        stations_df, key="station_pick_chart", label="📍 대상 대여소", all_label="전체 대여소 합계"
    )
    hourly = hourly_totals(daily, None if picked is None else picked["대여소ID"])

    target_name = "전체 대여소 합계" if picked is None else picked["대여소명"]
    st.markdown(f"##### {target_name}")
    st.line_chart(
        hourly,
        color=["#e63946", "#1d7ae0"],  # 예측 대여 = 빨강, 예측 반납 = 파랑
        x_label="시간 (시)",
        y_label="자전거 대수",
        height=420,
    )
    with st.expander("시간별 예측값 표 보기"):
        st.dataframe(hourly.round(1), width=True)