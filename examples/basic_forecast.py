from pathlib import Path
import os

# matplotlib이 사용자 홈 디렉터리에 캐시를 만들지 못하는 환경도 있어서,
# 프로젝트 내부의 로컬 캐시 디렉터리를 사용하도록 먼저 설정합니다.
os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from nixtla import NixtlaClient


# 결과 파일을 저장할 디렉터리를 미리 준비합니다.
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_sample_data() -> pd.DataFrame:
    # 외부 CSV 다운로드 없이도 바로 실험할 수 있도록
    # 작은 월별 샘플 시계열을 코드에서 직접 만듭니다.
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    values = [
        120, 128, 132, 129, 140, 145, 150, 148, 152, 160, 166, 172,
        175, 182, 188, 186, 194, 201, 208, 206, 214, 220, 228, 235,
        238, 244, 251, 249, 258, 266, 272, 270, 279, 286, 294, 301,
    ]
    return pd.DataFrame({"timestamp": dates, "value": values})


def main() -> None:
    # .env 파일이 있으면 환경 변수를 먼저 로드합니다.
    load_dotenv()

    # TimeGPT는 API Key가 필요하므로, 없으면 바로 안내하고 종료합니다.
    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key:
        raise ValueError("NIXTLA_API_KEY가 없습니다. .env 파일을 확인해주세요.")

    # 컬럼 의미:
    # - timestamp: 시계열 시점
    # - value: 예측할 대상 값
    df = build_sample_data()

    # 기본 클라이언트를 생성합니다.
    # api_key를 직접 넘겨도 되고, 현재처럼 환경 변수 기반으로 관리해도 됩니다.
    client = NixtlaClient(api_key=api_key)

    # 가장 기본적인 zero-shot forecast 예제입니다.
    # h=12 는 앞으로 12 스텝(여기서는 12개월) 예측을 뜻합니다.
    forecast_df = client.forecast(
        df=df,
        h=12,
        time_col="timestamp",
        target_col="value",
    )

    # 이후 분석이나 비교에 바로 쓸 수 있도록 CSV로 저장합니다.
    forecast_path = OUTPUT_DIR / "forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    # 원본 시계열과 예측 결과를 한 장의 그림으로 저장합니다.
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["value"], label="history")
    plt.plot(forecast_df["timestamp"], forecast_df["TimeGPT"], label="forecast")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "forecast.png")

    print(f"saved: {forecast_path}")
    print("saved: outputs/forecast.png")


if __name__ == "__main__":
    main()
