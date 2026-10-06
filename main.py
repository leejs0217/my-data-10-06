import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 모델 평가", layout="wide")
st.title("🌡️ 서울 연평균 기온 선형회귀 모델 평가 및 비교")

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

# 2. 데이터셋 분할
# 공통 테스트 데이터: 최근 20년 (2006 ~ 2025년)
test_df = df_filtered[(df_filtered["연도"] >= 2006) & (df_filtered["연도"] <= 2025)].copy()

# 학습 데이터 1: 최근 50년 (1956 ~ 2005년)
train_50y_df = df_filtered[(df_filtered["연도"] >= 1956) & (df_filtered["연도"] <= 2005)].copy()

# 학습 데이터 2: 최근 100년 (1906 ~ 2005년)
train_100y_df = df_filtered[(df_filtered["연도"] >= 1906) & (df_filtered["연도"] <= 2005)].copy()

# 3. 모델 학습 및 평가 함수
def train_and_evaluate(train_df, test_df):
    X_train = train_df["연도"].values
