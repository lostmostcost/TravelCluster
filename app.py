from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN

from preprocess import OUTPUT_CSV, preprocess_travel_data


EARTH_RADIUS_KM = 6371.0088
LATITUDE_COLUMN = "여행지 위도"
LONGITUDE_COLUMN = "여행지 경도"


@st.cache_data(show_spinner=False)
def load_processed_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        preprocess_travel_data(output_path=path)

    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def run_dbscan(df: pd.DataFrame, eps_km: float, min_samples: int) -> pd.DataFrame:
    coords_radians = np.radians(df[[LATITUDE_COLUMN, LONGITUDE_COLUMN]].to_numpy())
    labels = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit_predict(coords_radians)

    result = df.copy()
    result["분류"] = result["분류"].fillna("미분류").astype(str)
    result["cluster_id"] = labels
    result["is_noise"] = result["cluster_id"].eq(-1)
    result["eps_km"] = eps_km
    result["min_samples"] = min_samples
    result["cluster_label"] = np.where(
        result["is_noise"],
        "noise",
        "cluster " + result["cluster_id"].astype(str),
    )
    return result


def get_category_columns(categories: list[str]) -> list[str]:
    return [f"{category}_ratio" for category in categories]


def summarize_noise(clustered: pd.DataFrame, categories: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    noise_points = clustered[clustered["is_noise"]]
    total_count = len(clustered)
    noise_count = len(noise_points)
    noise_ratio = noise_count / total_count if total_count else 0.0

    if noise_points.empty:
        category_ratios = pd.Series(0.0, index=categories)
    else:
        category_ratios = noise_points["분류"].value_counts(normalize=True).reindex(categories, fill_value=0.0)

    noise_summary = pd.DataFrame(
        {
            "분류": categories,
            "노이즈 구성 비율": [float(category_ratios[category]) for category in categories],
            "노이즈 데이터 수": [
                int((noise_points["분류"] == category).sum()) for category in categories
            ],
        }
    ).sort_values(["노이즈 데이터 수", "분류"], ascending=[False, True])

    return {"noise_count": noise_count, "noise_ratio": noise_ratio}, noise_summary


def summarize_clusters(clustered: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    categories = sorted(clustered["분류"].fillna("미분류").astype(str).unique())
    category_columns = get_category_columns(categories)
    clustered_points = clustered[~clustered["is_noise"]]
    total_count = len(clustered)
    clustered_count = len(clustered_points)
    cluster_count = int(clustered_points["cluster_id"].nunique())
    included_ratio = clustered_count / total_count if total_count else 0.0
    category_averages = dict.fromkeys(category_columns, 0.0)

    if clustered_points.empty:
        summary = pd.DataFrame(columns=["cluster_id", "count", *category_columns])
    else:
        counts = (
            clustered_points.groupby("cluster_id")
            .size()
            .rename("count")
            .reset_index()
        )
        category_ratios = pd.crosstab(
            clustered_points["cluster_id"],
            clustered_points["분류"],
            normalize="index",
        )
        category_ratios = category_ratios.reindex(columns=categories, fill_value=0.0)
        category_ratios = category_ratios.rename(columns={category: f"{category}_ratio" for category in categories})
        summary = counts.merge(category_ratios.reset_index(), on="cluster_id", how="left")
        summary[category_columns] = summary[category_columns].fillna(0.0)
        summary = summary.sort_values(["count", "cluster_id"], ascending=[False, True])
        category_averages = {column: float(summary[column].mean()) for column in category_columns}

    metrics = {
        "cluster_count": cluster_count,
        "included_ratio": included_ratio,
    }
    category_summary = pd.DataFrame(
        {
            "분류": categories,
            "클러스터 평균 구성 비율": [category_averages[f"{category}_ratio"] for category in categories],
        }
    ).sort_values("클러스터 평균 구성 비율", ascending=False)
    noise_metrics, noise_summary = summarize_noise(clustered, categories)
    return metrics, summary, category_summary, noise_metrics, noise_summary


def build_map(clustered: pd.DataFrame):
    return px.scatter_mapbox(
        clustered,
        lat=LATITUDE_COLUMN,
        lon=LONGITUDE_COLUMN,
        color="cluster_label",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        hover_name="여행지명칭",
        hover_data={
            "주소": True,
            "지역구분": True,
            "분류": True,
            "태그": True,
            "cluster_id": True,
            LATITUDE_COLUMN: ":.5f",
            LONGITUDE_COLUMN: ":.5f",
            "cluster_label": False,
        },
        zoom=5.6,
        center={"lat": 36.2, "lon": 127.8},
        height=680,
        opacity=0.78,
    ).update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend_title_text="클러스터",
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return ("\ufeff" + df.to_csv(index=False)).encode("utf-8")


def sync_control_value(source_key: str, target_keys: tuple[str, ...]) -> None:
    for target_key in target_keys:
        st.session_state[target_key] = st.session_state[source_key]


def init_synced_control(control_key: str, default_value: float | int) -> tuple[str, str]:
    number_key = f"{control_key}_number"
    slider_key = f"{control_key}_slider"
    for key in (control_key, number_key, slider_key):
        st.session_state.setdefault(key, default_value)
    return number_key, slider_key


def synced_sidebar_float_control(
    label: str,
    control_key: str,
    min_value: float,
    max_value: float,
    default_value: float,
    step: float,
    help_text: str,
) -> float:
    number_key, slider_key = init_synced_control(control_key, default_value)
    st.sidebar.number_input(
        f"{label} 숫자 입력",
        min_value=min_value,
        max_value=max_value,
        step=step,
        format="%.1f",
        key=number_key,
        on_change=sync_control_value,
        args=(number_key, (slider_key, control_key)),
        help=help_text,
    )
    st.sidebar.slider(
        f"{label} 슬라이더",
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=slider_key,
        on_change=sync_control_value,
        args=(slider_key, (number_key, control_key)),
        help=help_text,
    )
    return float(st.session_state[control_key])


def synced_sidebar_int_control(
    label: str,
    control_key: str,
    min_value: int,
    max_value: int,
    default_value: int,
    step: int,
    help_text: str,
) -> int:
    number_key, slider_key = init_synced_control(control_key, default_value)
    st.sidebar.number_input(
        f"{label} 숫자 입력",
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=number_key,
        on_change=sync_control_value,
        args=(number_key, (slider_key, control_key)),
        help=help_text,
    )
    st.sidebar.slider(
        f"{label} 슬라이더",
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=slider_key,
        on_change=sync_control_value,
        args=(slider_key, (number_key, control_key)),
        help=help_text,
    )
    return int(st.session_state[control_key])


def main() -> None:
    st.set_page_config(page_title="여행지 DBSCAN 클러스터링", layout="wide")
    st.title("해양수산부 여행지 DBSCAN 클러스터링")
    st.caption("탐색 반경과 최소 군집 기준을 조절해 여행지 군집 변화를 지도와 지표로 확인합니다.")

    data = load_processed_data(str(OUTPUT_CSV))

    st.sidebar.header("DBSCAN 설정")
    eps_km = synced_sidebar_float_control(
        "탐색 반경 eps (km)",
        control_key="eps_km",
        min_value=0.1,
        max_value=50.0,
        default_value=3.0,
        step=0.1,
        help_text="한 군집으로 묶을 수 있는 최대 거리입니다.",
    )
    min_samples = synced_sidebar_int_control(
        "최소 군집 기준 min_samples",
        control_key="min_samples",
        min_value=2,
        max_value=50,
        default_value=5,
        step=1,
        help_text="핵심 지점이 되기 위해 반경 안에 필요한 최소 여행지 수입니다.",
    )

    st.sidebar.divider()
    st.sidebar.write(f"전처리 데이터: `{OUTPUT_CSV.name}`")
    st.sidebar.write(f"분석 대상: `{len(data):,}`개 여행지")

    with st.spinner("DBSCAN 군집을 계산하는 중입니다..."):
        clustered = run_dbscan(data, eps_km, min_samples)
        metrics, cluster_summary, category_summary, noise_metrics, noise_summary = summarize_clusters(clustered)

    metric_columns = st.columns(5)
    metric_columns[0].metric("총 클러스터 수", f"{metrics['cluster_count']:,}개")
    metric_columns[1].metric("클러스터 포함 비율", f"{metrics['included_ratio']:.1%}")
    metric_columns[2].metric("노이즈 데이터 수", f"{noise_metrics['noise_count']:,}개")
    metric_columns[3].metric("노이즈 비율", f"{noise_metrics['noise_ratio']:.1%}")
    metric_columns[4].metric("분류 항목 수", f"{len(category_summary):,}개")

    st.subheader("분류별 클러스터 평균 구성 비율")
    st.dataframe(
        category_summary.assign(
            **{
                "클러스터 평균 구성 비율": lambda frame: frame["클러스터 평균 구성 비율"].map("{:.1%}".format)
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("대한민국 지도 기반 클러스터 시각화")
    st.plotly_chart(build_map(clustered), use_container_width=True)

    st.subheader("클러스터 요약")
    ratio_columns = [column for column in cluster_summary.columns if column.endswith("_ratio")]
    formatted_cluster_summary = cluster_summary.copy()
    for column in ratio_columns:
        formatted_cluster_summary[column] = formatted_cluster_summary[column].map("{:.1%}".format)

    st.dataframe(
        formatted_cluster_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("클러스터 미포함 데이터 통계 (cluster_id = -1)")
    st.caption("DBSCAN에서 `cluster_id = -1`은 설정한 반경과 최소 기준으로 어떤 클러스터에도 포함되지 못한 노이즈 데이터입니다.")
    st.dataframe(
        noise_summary.assign(
            **{"노이즈 구성 비율": lambda frame: frame["노이즈 구성 비율"].map("{:.1%}".format)}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("현재 설정 결과 저장")
    download_columns = [
        "여행지 경도",
        "여행지 위도",
        "주소",
        "지역구분",
        "여행지명칭",
        "분류",
        "태그",
        "cluster_id",
        "is_noise",
        "eps_km",
        "min_samples",
    ]
    file_name = f"clustered_travel_destinations_eps_{eps_km:.1f}_min_{min_samples}.csv"
    st.download_button(
        "현재 클러스터 결과 CSV 다운로드",
        data=to_csv_bytes(clustered[download_columns]),
        file_name=file_name,
        mime="text/csv",
    )

    with st.expander("결과 데이터 미리보기"):
        st.dataframe(clustered[download_columns].head(200), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

