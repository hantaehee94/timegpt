# time-gpt

CSV 시계열에서 원하는 시작 지점과 입력 길이를 고른 뒤, TimeGPT 예측값을 실제값과 비교해 정확도를 확인하는 최소 예제입니다.

참고한 TimeGPT 공식 문서:

- [Quickstart](https://www.nixtla.io/docs/forecasting/timegpt_quickstart): `NixtlaClient.forecast(df, h, freq, time_col, target_col)` 사용
- [Data Requirements](https://www.nixtla.io/docs/data_requirements/data_requirements): timestamp, target, optional `unique_id` 컬럼 형식
- [Evaluation Pipeline](https://www.nixtla.io/docs/forecasting/evaluation/evaluation_utilsforecast): forecast 결과와 test 데이터를 합쳐 평가

## 1. 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. API Key 설정

```bash
cp .env.example .env
```

`.env` 파일에 `NIXTLA_API_KEY` 값을 넣어주세요.

## 3. CSV 형식

기본 예시는 `inputs/sample_series.csv`입니다.

```csv
unique_id,timestamp,value
sample,2021-01-01,120
sample,2021-02-01,128
```

- `timestamp`: 시계열 시점
- `value`: 예측할 값
- `unique_id`: 여러 시계열을 한 CSV에 넣을 때 쓰는 선택 컬럼

핵심 구조는 아래만 보면 됩니다.

```text
.
├── run_forecast_eval.py
├── inputs/
│   └── sample_series.csv
├── outputs/
├── requirements.txt
└── README.md
```

## 4. 실행

예: `2022-01-01`부터 36개 timestamp를 입력하고, 바로 다음 12개 timestamp를 예측해 실제값과 비교합니다.

```bash
python run_forecast_eval.py \
  --csv inputs/sample_series.csv \
  --start 2022-01-01 \
  --input-steps 36 \
  --horizon 12 \
  --freq MS
```

`--start`에는 timestamp 대신 행 번호도 넣을 수 있습니다. 예: `--start 12`

API 호출 없이 구간만 확인하려면:

```bash
python run_forecast_eval.py --start 2022-01-01 --input-steps 36 --horizon 12 --dry-run
```

## 5. 결과

실행이 끝나면 아래 파일이 생성됩니다.

- `outputs/forecast_comparison.csv`: timestamp별 실제값, 예측값, 오차
- `outputs/metrics.csv`: MAE, RMSE, MAPE, WAPE, accuracy
- `outputs/forecast_comparison.png`: 입력 구간, 실제값, 예측값 그래프

다른 CSV 컬럼명을 쓰는 경우:

```bash
python run_forecast_eval.py \
  --csv inputs/my_data.csv \
  --time-col date \
  --target-col sales \
  --id-col store_id \
  --series-id store_001 \
  --start 2024-01-01 \
  --input-steps 90 \
  --horizon 14 \
  --freq D
```
