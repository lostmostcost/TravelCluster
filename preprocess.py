from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_CSV = BASE_DIR / "여행지정보" / "해양수산부_여행지 정보_20250709.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_CSV = OUTPUT_DIR / "processed_travel_destinations.csv"

KEEP_COLUMNS = [
    "여행지 경도",
    "여행지 위도",
    "주소",
    "지역구분",
    "여행지명칭",
    "분류",
    "태그",
]


def read_source_csv(path: Path = SOURCE_CSV) -> pd.DataFrame:
    """Read the source CSV while handling common Korean CSV encodings."""
    encodings = ("utf-8-sig", "utf-8", "cp949")
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"CSV 파일을 읽을 수 없습니다: {path}")


def preprocess_travel_data(
    source_path: Path = SOURCE_CSV,
    output_path: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    df = read_source_csv(source_path)

    missing_columns = [column for column in KEEP_COLUMNS if column not in df.columns]
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise ValueError(f"원본 CSV에 필요한 컬럼이 없습니다: {joined_columns}")

    processed = df.loc[:, KEEP_COLUMNS].copy()
    processed["분류"] = processed["분류"].fillna("미분류").astype(str).str.strip()
    processed["분류"] = processed["분류"].replace({"식당": "음식점"})
    processed["여행지 경도"] = pd.to_numeric(processed["여행지 경도"], errors="coerce")
    processed["여행지 위도"] = pd.to_numeric(processed["여행지 위도"], errors="coerce")
    processed = processed.dropna(subset=["여행지 경도", "여행지 위도"])

    processed = processed[
        processed["여행지 경도"].between(124.0, 132.5)
        & processed["여행지 위도"].between(32.0, 39.5)
    ]
    processed = processed.drop_duplicates().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False, encoding="utf-8-sig")
    return processed


def main() -> None:
    processed = preprocess_travel_data()
    print(f"전처리 완료: {OUTPUT_CSV}")
    print(f"행 수: {len(processed):,}")
    print(f"컬럼: {', '.join(processed.columns)}")


if __name__ == "__main__":
    main()

