# 해양수산부 여행지 DBSCAN 클러스터링 웹앱 작업 지시서

## 목표

`여행지정보/해양수산부_여행지 정보_20250709.csv` 데이터를 전처리한 뒤, DBSCAN 군집화 파라미터를 웹에서 실시간으로 조절하며 대한민국 지도 위에 클러스터 결과를 시각화한다. 사용자는 현재 파라미터 상태의 클러스터 결과를 CSV로 다운로드할 수 있어야 한다.

## 입력 데이터

- 원본 파일: `여행지정보/해양수산부_여행지 정보_20250709.csv`
- 원본 CSV는 수정하지 않는다.

전처리 후 유지할 컬럼:

- `여행지 경도`
- `여행지 위도`
- `주소`
- `지역구분`
- `여행지명칭`
- `분류`
- `태그`

## 산출물

- 전처리 CSV: `output/processed_travel_destinations.csv`
- Streamlit 웹앱: `app.py`
- 전처리 스크립트: `preprocess.py`
- 설치 의존성: `requirements.txt`
- 실행 안내 문서: `README.md`
- 다운로드 CSV: 현재 DBSCAN 설정에 따른 `cluster_id`, `is_noise`, `eps_km`, `min_samples` 컬럼 포함

## 구현 요구사항

1. 원본 CSV에서 지정된 7개 컬럼만 남긴다.
2. `여행지 경도`, `여행지 위도`를 숫자로 변환하고, 좌표가 없는 행은 제거한다.
3. 완전히 동일한 중복 행은 제거하되, 같은 좌표나 같은 여행지명만으로는 제거하지 않는다.
4. DBSCAN은 위도/경도 좌표를 라디안으로 변환한 뒤 Haversine 거리 기준으로 수행한다.
5. 사용자는 웹 사이드바에서 `eps_km`와 `min_samples`를 슬라이더로 조절할 수 있어야 한다.
6. 슬라이더 변경 시 다음 지표를 실시간으로 갱신한다.
   - 생성된 총 클러스터 개수
   - 전체 여행지의 클러스터 포함 비율
   - 클러스터 내 식당/숙박/체험 평균 구성 비율
7. 대한민국 지도 위에 클러스터별 색상을 다르게 표시하고, 노이즈 데이터는 별도 색상으로 표시한다.
8. 지도 점에 마우스를 올리면 `여행지명칭`, `주소`, `지역구분`, `분류`, `태그`, `cluster_id`를 확인할 수 있어야 한다.
9. 현재 슬라이더 상태의 클러스터 결과를 CSV로 다운로드할 수 있는 버튼을 제공한다.

## 권장 기술

- 데이터 처리: `pandas`, `numpy`
- 군집화: `scikit-learn`
- 웹 UI: `streamlit`
- 지도 시각화: `plotly`
- CSV 다운로드: Streamlit `download_button`

## 권장 파일 구조

```text
Cluster/
├── 여행지정보/
│   └── 해양수산부_여행지 정보_20250709.csv
├── output/
│   └── processed_travel_destinations.csv
├── app.py
├── preprocess.py
├── requirements.txt
└── README.md
```

## 실행 방법

```bash
pip install -r requirements.txt
python preprocess.py
streamlit run app.py
```

## 검증 기준

- 전처리 CSV가 지정된 7개 컬럼만 포함한다.
- 위도/경도 결측 또는 비정상 행이 제거된다.
- `eps_km`, `min_samples` 슬라이더 변경 시 통계와 지도가 갱신된다.
- 총 클러스터 개수와 클러스터 포함 비율은 노이즈 데이터를 제외하고 계산된다.
- 다운로드 CSV에 `cluster_id`, `is_noise`, `eps_km`, `min_samples`가 포함된다.
- 원본 CSV 파일은 변경되지 않는다.

