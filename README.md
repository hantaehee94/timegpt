# time-gpt

`TimeGPT` 실험을 빠르게 시작하기 위한 최소 Python 환경입니다.

## 1. 가상환경 생성

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. API Key 설정

```bash
cp .env.example .env
```

`.env` 파일에 `NIXTLA_API_KEY` 값을 넣어주세요.

## 4. 기본 예제 실행

```bash
python examples/basic_forecast.py
```

실행이 끝나면 아래 파일이 생성됩니다.

- `outputs/forecast.csv`
- `outputs/forecast.png`
