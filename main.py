import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# 1. Page Configuration (브라우저 탭 제목 및 아이콘 설정)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 화면 상단 제목 및 아이콘
st.title("🎬 영화 유형 나누기")
st.markdown("클러스터링(K-Means) 기법을 활용하여 영화를 속성별로 유형화합니다.")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    total_count = len(df)
    
    # 4개 속성 중 결측값이 있거나 first_week_audi가 0인 영화 제외
    df_valid = df.dropna(subset=['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']).copy()
    df_valid = df_valid[df_valid['first_week_audi'] > 0].copy()
    
    # 롱런 지수 생성: 누적 관객 / 첫 주 관객 (20 초과 시 20으로 클리핑)
    df_valid['long_run_idx'] = df_valid['total_audi'] / df_valid['first_week_audi']
    df_valid['long_run_idx'] = df_valid['long_run_idx'].clip(upper=20)
    
    # 상용로그 취하기 (first_scrn, total_audi)
    df_valid['log_first_scrn'] = np.log10(df_valid['first_scrn'])
    df_valid['log_total_audi'] = np.log10(df_valid['total_audi'])
    
    used_count = len(df_valid)
    return df_valid, total_count, used_count

df_work, total_count, used_count = load_and_preprocess_data()

# 전체 편수와 묶은 편수를 한 줄로 표시
st.info(f"📊 **전체 영화 수:** {total_count:,}편 | **유형 분석에 사용된 영화 수:** {used_count:,}편")

# 3. 묶는 데 사용할 속성 선택 (다중 선택, 기본값 4개 전체)
st.subheader("⚙️ 클러스터링 속성 선택")

feature_dict = {
    "스크린 수 (log10)": "log_first_scrn",
    "누적 관객 수 (log10)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_idx"
}

selected_feature_labels = st.multiselect(
    "묶는 데 사용할 속성을 2개 이상 선택하세요 (기본값: 4개 모두 선택):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys())
)

if len(selected_feature_labels) < 2:
    st.warning("⚠️ 클러스터링을 진행하려면 최소 2개 이상의 속성을 선택해야 합니다.")
    st.stop()

selected_feature_cols = [feature_dict[label] for label in selected_feature_labels]

# 4. 표준화(StandardScaler) 및 K-Means 클러스터링 (3개 묶음, 난수 고정 random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_work[selected_feature_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_work['cluster_raw'] = clusters

# 누적 관객 평균(total_audi)이 큰 순서대로 ㉮, ㉯, ㉰ 이름 붙이기
cluster_avg_audi = df_work.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
label_mapping = {}
labels_symbols = ['㉮', '㉯', '㉰']

for rank, (cluster_id, _) in enumerate(cluster_avg_audi.items()):
    label_mapping[cluster_id] = labels_symbols[rank]

df_work['cluster'] = df_work['cluster_raw'].map(label_mapping)

st.divider()

# 5. 2차원 산점도 (Plotly)
st.subheader("📈 2차원 산점도")

col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_2d_label = st.selectbox("가로축 (X축) 속성:", options=selected_feature_labels, index=0, key="2d_x")
with col2d_2:
    y_2d_label = st.selectbox("세로축 (Y축) 속성:", options=selected_feature_labels, index=min(1, len(selected_feature_labels)-1), key="2d_y")

fig_2d = px.scatter(
    df_work,
    x=feature_dict[x_2d_label],
    y=feature_dict[y_2d_label],
    color="cluster",
    color_discrete_map={'㉮': '#EF553B', '㉯': '#00CC96', '㉰': '#636EFA'},
    hover_name="movieNm",
    labels={
        feature_dict[x_2d_label]: x_2d_label,
        feature_dict[y_2d_label]: y_2d_label,
        "cluster": "유형"
    },
    category_orders={"cluster": ['㉮', '㉯', '㉰']},
    title=f"2차원 산점도 ({x_2d_label} vs {y_2d_label})"
)
fig_2d.update_traces(marker=dict(size=6, opacity=0.8))
st.plotly_chart(fig_2d, use_container_width=True)

# 6. 3차원 산점도 (Plotly)
st.subheader("🧊 3차원 산점도")

if len(selected_feature_labels) < 3:
    st.info("💡 3차원 산점도를 그리려면 묶는 데 사용하는 속성을 3개 이상 선택하세요.")
else:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        x_3d_label = st.selectbox("X축 속성:", options=selected_feature_labels, index=0, key="3d_x")
    with col3d_2:
        y_3d_label = st.selectbox("Y축 속성:", options=selected_feature_labels, index=1, key="3d_y")
    with col3d_3:
        z_3d_label = st.selectbox("Z축 속성:", options=selected_feature_labels, index=2, key="3d_z")

    fig_3d = px.scatter_3d(
        df_work,
        x=feature_dict[x_3d_label],
        y=feature_dict[y_3d_label],
        z=feature_dict[z_3d_label],
        color="cluster",
        color_discrete_map={'㉮': '#EF553B', '㉯': '#00CC96', '㉰': '#636EFA'},
        hover_name="movieNm",
        labels={
            feature_dict[x_3d_label]: x_3d_label,
            feature_dict[y_3d_label]: y_3d_label,
            feature_dict[z_3d_label]: z_3d_label,
            "cluster": "유형"
        },
        category_orders={"cluster": ['㉮', '㉯', '㉰']},
        title=f"3차원 산점도 ({x_3d_label} / {y_3d_label} / {z_3d_label})"
    )
    fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# 7. 묶음별 편수 및 4가지 속성의 평균(원래 단위) 표 출력
st.subheader("📊 유형별 편수 및 속성 평균 (원래 단위)")

summary_df = df_work.groupby('cluster').agg(
    영화편수=('movieCd', 'count'),
    스크린수_평균=('first_scrn', 'mean'),
    누적관객_평균=('total_audi', 'mean'),
    십위권일수_평균=('days_in_top10', 'mean'),
    롱런지수_평균=('long_run_idx', 'mean')
).reindex(['㉮', '㉯', '㉰']).reset_index().rename(columns={'cluster': '유형'})

# 표시 형식 가공
summary_df_display = summary_df.copy()
summary_df_display['영화편수'] = summary_df_display['영화편수'].apply(lambda x: f"{x:,}편")
summary_df_display['스크린수_평균'] = summary_df_display['스크린수_평균'].apply(lambda x: f"{x:,.1f}개")
summary_df_display['누적관객_평균'] = summary_df_display['누적관객_평균'].apply(lambda x: f"{x:,.0f}명")
summary_df_display['십위권일수_평균'] = summary_df_display['십위권일수_평균'].apply(lambda x: f"{x:,.1f}일")
summary_df_display['롱런지수_평균'] = summary_df_display['롱런지수_평균'].apply(lambda x: f"{x:,.2f}")

st.dataframe(summary_df_display, use_container_width=True)

# 8. 묶음별 누적 관객 수가 많은 순서로 영화 5편 제목 표시
st.subheader("🏆 유형별 누적 관객 Top 5 영화")

col_a, col_b, col_c = st.columns(3)
cols_list = [col_a, col_b, col_c]

for idx, symbol in enumerate(['㉮', '㉯', '㉰']):
    with cols_list[idx]:
        st.markdown(f"### 유형 {symbol}")
        top5 = df_work[df_work['cluster'] == symbol].sort_values(by='total_audi', ascending=False).head(5)
        
        top5_display = top5[['movieNm', 'total_audi']].reset_index(drop=True)
        top5_display.index = top5_display.index + 1
        top5_display.columns = ['영화 제목', '누적 관객 수']
        top5_display['누적 관객 수'] = top5_display['누적 관객 수'].apply(lambda x: f"{x:,}명")
        
        st.table(top5_display)
