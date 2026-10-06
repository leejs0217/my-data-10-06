import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="서울 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

DATA_END_YEAR = 2025
MIN_DAYS = 300

# 공통 테스트 기간
TEST_START = 2006
TEST_END = 2025

# 훈련 기간
TRAIN_50_START = 1956
TRAIN_50_END = 2005

TRAIN_100_START = 1906
TRAIN_100_END = 2005


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 유효한 데이터만 사용
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 연도
    df["연도"] = df["날짜"].dt.year

    # 2025년까지 사용
    df = df[
        df["연도"] <= DATA_END_YEAR
    ].copy()

    return df


# =========================================================
# 연평균기온 계산
# =========================================================

@st.cache_data
def make_yearly_data(df):

    # 연도별 관측일수
    days = (
        df.groupby("연도")
        .size()
        .reset_index(name="관측일수")
    )

    # 연도별 평균기온
    temps = (
        df.groupby("연도")["평균기온"]
        .mean()
        .reset_index(name="연평균기온")
    )

    yearly = pd.merge(
        temps,
        days,
        on="연도",
        how="inner"
    )

    # 관측일 300일 이상인 연도만 사용
    yearly = yearly[
        yearly["관측일수"] >= MIN_DAYS
    ].copy()

    yearly = yearly.sort_values(
        "연도"
    ).reset_index(drop=True)

    return yearly


# =========================================================
# 선형회귀
# =========================================================

def train_regression(train_df):

    # 연도를 독립변수로 사용
    X = train_df[["연도"]]
    y = train_df["연평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


# =========================================================
# 평가
# =========================================================

def evaluate_model(model, test_df):

    X_test = test_df[["연도"]]
    y_test = test_df["연평균기온"]

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    return mae, mse, r2, y_pred


# =========================================================
# 앱
# =========================================================

st.title("🌡️ 서울 연평균기온 선형회귀 분석")

st.write(
    """
    서울 일별 평균기온 데이터를 연평균기온으로 변환한 뒤,
    과거 데이터를 이용해 선형회귀 모델을 학습하고
    2006~2025년을 공통 테스트 데이터로 사용하여
    예측 성능을 비교합니다.
    """
)


# =========================================================
# 데이터 준비
# =========================================================

try:

    raw_data = load_data()
    yearly = make_yearly_data(raw_data)

except Exception as e:

    st.error(
        f"데이터를 불러오는 중 오류가 발생했습니다.\n\n{e}"
    )

    st.stop()


# =========================================================
# 훈련 / 테스트 데이터 분리
# =========================================================

train_50 = yearly[
    (yearly["연도"] >= TRAIN_50_START)
    & (yearly["연도"] <= TRAIN_50_END)
].copy()


train_100 = yearly[
    (yearly["연도"] >= TRAIN_100_START)
    & (yearly["연도"] <= TRAIN_100_END)
].copy()


test = yearly[
    (yearly["연도"] >= TEST_START)
    & (yearly["연도"] <= TEST_END)
].copy()


# 전체 기간 모델
full_data = yearly[
    (yearly["연도"] >= 1908)
    & (yearly["연도"] <= DATA_END_YEAR)
].copy()


# =========================================================
# 모델 학습
# =========================================================

model_50 = train_regression(train_50)
model_100 = train_regression(train_100)
model_full = train_regression(full_data)


# =========================================================
# 테스트 성능 평가
# =========================================================

mae_50, mse_50, r2_50, pred_50 = evaluate_model(
    model_50,
    test
)

mae_100, mse_100, r2_100, pred_100 = evaluate_model(
    model_100,
    test
)

# 참고용: 전체 기간 모델
mae_full, mse_full, r2_full, pred_full = evaluate_model(
    model_full,
    test
)


# =========================================================
# 회귀선 정보
# =========================================================

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_

slope_full = model_full.coef_[0]
intercept_full = model_full.intercept_


# =========================================================
# 제목
# =========================================================

st.header("1. 훈련 데이터 구성")


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "50년 훈련",
        f"{len(train_50)}개 연도"
    )

    st.caption(
        f"{TRAIN_50_START}~{TRAIN_50_END}"
    )


with col2:

    st.metric(
        "100년 훈련",
        f"{len(train_100)}개 연도"
    )

    st.caption(
        f"{TRAIN_100_START}~{TRAIN_100_END}"
    )


with col3:

    st.metric(
        "공통 테스트",
        f"{len(test)}개 연도"
    )

    st.caption(
        f"{TEST_START}~{TEST_END}"
    )


st.info(
    "1906~2005년을 요청했지만 실제 데이터가 존재하지 않거나 "
    "관측일수가 300일 미만인 연도는 자동으로 제외됩니다."
)


# =========================================================
# 회귀선 기울기 비교
# =========================================================

st.header("2. 회귀선 기울기 비교")


slope_table = pd.DataFrame(
    {
        "모델": [
            "최근 50년",
            "최근 100년",
            "전체 기간"
        ],
        "훈련기간": [
            "1956~2005",
            "1906~2005",
            "1908~2025"
        ],
        "기울기 (℃/년)": [
            slope_50,
            slope_100,
            slope_full
        ],
        "절편": [
            intercept_50,
            intercept_100,
            intercept_full
        ]
    }
)


st.dataframe(
    slope_table.style.format(
        {
            "기울기 (℃/년)": "{:.5f}",
            "절편": "{:.3f}"
        }
    ),
    width="stretch"
)


st.markdown(
    f"""
**기울기 해석**

- 최근 50년 회귀선: 연평균기온이 매년 약 **{slope_50:.4f}℃** 변화
- 최근 100년 회귀선: 연평균기온이 매년 약 **{slope_100:.4f}℃** 변화
- 전체 기간 회귀선: 연평균기온이 매년 약 **{slope_full:.4f}℃** 변화

기울기가 클수록 시간의 흐름에 따른 연평균기온 상승 추세가 더 가파른 것입니다.
"""
)


# =========================================================
# 회귀선 시각화
# =========================================================

st.header("3. 50년 vs 100년 회귀선")


line_years = np.arange(
    min(1906, yearly["연도"].min()),
    DATA_END_YEAR + 1
)

pred_line_50 = model_50.predict(
    line_years.reshape(-1, 1)
)

pred_line_100 = model_100.predict(
    line_years.reshape(-1, 1)
)


fig_lines = go.Figure()


# 실제 연평균기온
fig_lines.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6),
        hovertemplate=(
            "%{x}년<br>"
            "실제: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 회귀선
fig_lines.add_trace(
    go.Scatter(
        x=line_years,
        y=pred_line_50,
        mode="lines",
        name="1956~2005 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "%{x}년<br>"
            "50년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 회귀선
fig_lines.add_trace(
    go.Scatter(
        x=line_years,
        y=pred_line_100,
        mode="lines",
        name="1906~2005 회귀선",
        line=dict(width=3, dash="dash"),
        hovertemplate=(
            "%{x}년<br>"
            "100년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 테스트 시작점
fig_lines.add_vline(
    x=TEST_START,
    line_dash="dot",
    annotation_text="테스트 시작: 2006년"
)


fig_lines.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=600,
    hovermode="x unified"
)


st.plotly_chart(
    fig_lines,
    width="stretch"
)


# =========================================================
# 모델 성능 비교
# =========================================================

st.header("4. 공통 테스트 데이터 예측 성능")

st.write(
    """
    세 모델 모두 동일하게 **2006~2025년** 데이터를 테스트 데이터로
    사용합니다. 따라서 50년 모델과 100년 모델을 공정하게 비교할 수 있습니다.
    """
)


performance = pd.DataFrame(
    {
        "모델": [
            "최근 50년 학습",
            "최근 100년 학습",
            "전체 기간 학습*"
        ],
        "훈련기간": [
            "1956~2005",
            "1906~2005",
            "1908~2025"
        ],
        "MAE (℃)": [
            mae_50,
            mae_100,
            mae_full
        ],
        "MSE (℃²)": [
            mse_50,
            mse_100,
            mse_full
        ],
        "R²": [
            r2_50,
            r2_100,
            r2_full
        ]
    }
)


st.dataframe(
    performance.style.format(
        {
            "MAE (℃)": "{:.4f}",
            "MSE (℃²)": "{:.4f}",
            "R²": "{:.4f}"
        }
    ),
    width="stretch"
)


st.caption(
    "* 전체 기간 모델은 2006~2025년을 학습에도 포함하므로 "
    "엄밀한 의미의 독립적인 테스트 성능 비교용이 아니라 참고용입니다."
)


# =========================================================
# 실제값 vs 예측값
# =========================================================

st.header("5. 2006~2025년 실제값 vs 예측값")


test_result = test.copy()

test_result["50년 예측"] = pred_50
test_result["100년 예측"] = pred_100


fig_test = go.Figure()


# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        line=dict(width=3),
        hovertemplate=(
            "%{x}년<br>"
            "실제: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 모델 예측
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["50년 예측"],
        mode="lines+markers",
        name="50년 학습 모델",
        hovertemplate=(
            "%{x}년<br>"
            "50년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 모델 예측
fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["100년 예측"],
        mode="lines+markers",
        name="100년 학습 모델",
        hovertemplate=(
            "%{x}년<br>"
            "100년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=2
    ),
    height=600,
    hovermode="x unified"
)


st.plotly_chart(
    fig_test,
    width="stretch"
)


# =========================================================
# 성능 차이 분석
# =========================================================

st.header("6. 50년 vs 100년 모델 비교")


mae_diff = mae_50 - mae_100
mse_diff = mse_50 - mse_100
r2_diff = r2_50 - r2_100


comparison_col1, comparison_col2 = st.columns(2)


with comparison_col1:

    st.subheader("오차 지표")

    st.write(
        f"MAE 차이 (50년 - 100년): "
        f"**{mae_diff:+.4f}℃**"
    )

    st.write(
        f"MSE 차이 (50년 - 100년): "
        f"**{mse_diff:+.4f}℃²**"
    )


with comparison_col2:

    st.subheader("설명력")

    st.write(
        f"R² 차이 (50년 - 100년): "
        f"**{r2_diff:+.4f}**"
    )


if mae_50 < mae_100 and r2_50 > r2_100:

    st.success(
        "최근 50년 모델이 테스트 데이터에서 "
        "MAE와 R² 모두 더 좋은 성능을 보입니다."
    )

elif mae_100 < mae_50 and r2_100 > r2_50:

    st.success(
        "최근 100년 모델이 테스트 데이터에서 "
        "MAE와 R² 모두 더 좋은 성능을 보입니다."
    )

else:

    st.info(
        "50년 모델과 100년 모델의 우위가 평가 지표에 따라 다릅니다. "
        "MAE·MSE는 낮을수록, R²는 높을수록 좋습니다."
    )


# =========================================================
# 모델 해석
# =========================================================

st.header("7. 결과 해석")

st.markdown(
    f"""
### MAE
평균적으로 실제 연평균기온에서 얼마나 벗어났는지를 나타냅니다.

- 50년 모델: **{mae_50:.4f}℃**
- 100년 모델: **{mae_100:.4f}℃**

낮을수록 좋습니다.

### MSE
예측 오차를 제곱해서 평균낸 값입니다.
큰 오차에 더 큰 벌점을 줍니다.

- 50년 모델: **{mse_50:.4f}**
- 100년 모델: **{mse_100:.4f}**

낮을수록 좋습니다.

### R²
실제 연평균기온의 변동을 회귀모델이 얼마나 설명하는지를 나타냅니다.

- 50년 모델: **{r2_50:.4f}**
- 100년 모델: **{r2_100:.4f}**

일반적으로 높을수록 좋습니다.

### 기울기
- 50년 모델: **{slope_50:.5f}℃/년**
- 100년 모델: **{slope_100:.5f}℃/년**

따라서 두 모델의 기울기 차이는

**{abs(slope_50 - slope_100):.5f}℃/년**

입니다.
"""
)


# =========================================================
# 데이터 표
# =========================================================

with st.expander("📋 연도별 연평균기온 데이터 보기"):

    st.dataframe(
        yearly,
        width="stretch"
    )
