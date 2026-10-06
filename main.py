import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 모델 평가", layout="wide")
st.title("🌡️ 학습 기간별 선형회귀 모델 예측 성능 비교 (MAE, MSE, R²)")

# 1. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"].str.strip())
    df["연도"] = df["날짜"].dt.year
    
    yearly_summary = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일 300일 이상 유효 데이터 필터링 (2025년 이하)
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    return filtered_df

df_filtered = load_and_process_data()

# 2. 데이터셋 분할 (학습 데이터 2종 & 공통 테스트 데이터 1종)
# 테스트 데이터: 최근 20년 (2006 ~ 2025년)
test_df = df_filtered[(df_filtered["연도"] >= 2006) & (df_filtered["연도"] <= 2025)].copy()

# 학습 데이터 1: 최근 50년 (1956 ~ 2005년)
train_50y_df = df_filtered[(df_filtered["연도"] >= 1956) & (df_filtered["연도"] <= 2005)].copy()

# 학습 데이터 2: 최근 100년 (1906 ~ 2005년)
train_100y_df = df_filtered[(df_filtered["연도"] >= 1906) & (df_filtered["연도"] <= 2005)].copy()

# 3. 모델 학습 및 테스트 데이터 평가 함수
def train_and_evaluate(train_df, test_df):
    # 독립변수(X): 연도, 종속변수(y): 연평균기온
    X_train = train_df["연도"].values
    y_train = train_df["연평균기온"].values
    
    X_test = test_df["연도"].values
    y_test = test_df["연평균기온"].values
    
    # 1차 선형 회귀 적합
    slope, intercept = np.polyfit(X_train, y_train, 1)
    
    # 테스트 데이터에 대한 예측값 계산
    y_pred_test = slope * X_test + intercept
    
    # 평가지표 계산
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    return slope, intercept, y_pred_test, mae, mse, r2

# 모델 실행
slope_50, intercept_50, pred_50, mae_50, mse_50, r2_50 = train_and_evaluate(train_50y_df, test_df)
slope_100, intercept_100, pred_100, mae_100, mse_100, r2_100 = train_and_evaluate(train_100y_df, test_df)

# 전체 데이터 기준 모델 (참고용)
slope_all, intercept_all = np.polyfit(df_filtered["연도"], df_filtered["연평균기온"], 1)

# 4. 결과 출력: 평가지표 및 기울기 비교 표
st.subheader("📋 모델별 기울기 및 테스트 데이터(2006~2025년) 예측 성능 평가")

metrics_data = {
    "구분": ["최근 50년 학습 모델 (1956~2005)", "최근 100년 학습 모델 (1906~2005)"],
    "학습 기간": ["1956년 ~ 2005년 (50개 해)", "1906년 ~ 2005년 (100개 해)"],
    "기울기 (°C/년)": [f"{slope_50:+.4f}", f"{slope_100:+.4f}"],
    "100년당 상승률 (°C/100년)": [f"{slope_50 * 100:+.2f} °C", f"{slope_100 * 100:+.2f} °C"],
    "MAE (°C)": [f"{mae_50:.4f}", f"{mae_100:.4f}"],
    "MSE (°C²)": [f"{mse_50:.4f}", f"{mse_100:.4f}"],
    "R² (결정계수)": [f"{r2_50:.4f}", f"{r2_100:.4f}"]
}

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# 5. 핵심 지표 Metric 표시
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="⚡ 50년 모델 100년당 상승률",
        value=f"{slope_50 * 100:+.2f} °C / 100년",
        delta=f"MAE 오차: {mae_50:.2f} °C"
    )

with col2:
    st.metric(
        label="🌐 100년 모델 100년당 상승률",
        value=f"{slope_100 * 100:+.2f} °C / 100년",
        delta=f"MAE 오차: {mae_100:.2f} °C",
        delta_color="inverse"
    )

st.markdown("---")

# 6. Plotly 시각화 (학습 구간, 테스트 구간, 회귀 외삽선)
st.subheader("📈 학습 구간 회귀선과 테스트 데이터(2006~2025) 외삽 예측 비교")

line_years = np.arange(1900, 2030)
line_y_50 = slope_50 * line_years + intercept_50
line_y_100 = slope_100 * line_years + intercept_100

fig = go.Figure()

# 1) 전체 실제 데이터
fig.add_trace(go.Scatter(
    x=df_filtered["연도"],
    y=df_filtered["연평균기온"],
    mode="markers",
    name="실제 기온 (1908~2025)",
    marker=dict(color="lightslategrey", size=6, opacity=0.5),
    hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>"
))

# 2) 테스트 데이터 강조 (2006~2025)
fig.add_trace(go.Scatter(
    x=test_df["연도"],
    y=test_df["연평균기온"],
    mode="markers",
    name="테스트 데이터 (2006~2025)",
    marker=dict(color="crimson", size=9, symbol="diamond"),
    hovertemplate="테스트 %{x}년: %{y:.2f}°C<extra></extra>"
))

# 3) 최근 50년 모델 회귀선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_y_50,
    mode="lines",
    name=f"50년 학습 회귀선 ({slope_50*100:+.2f}°C/100년)",
    line=dict(color="darkorange", width=2.5),
    hovertemplate="%{x}년 예측(50년모델): %{y:.2f}°C<extra></extra>"
))

# 4) 최근 100년 모델 회귀선
fig.add_trace(go.Scatter(
    x=line_years,
    y=line_y_100,
    mode="lines",
    name=f"100년 학습 회귀선 ({slope_100*100:+.2f}°C/100년)",
    line=dict(color="royalblue", width=2.5, dash="dash"),
    hovertemplate="%{x}년 예측(100년모델): %{y:.2f}°C<extra></extra>"
))

fig.update_layout(
    title="서울 기온 예측: 50년 학습 vs 100년 학습
