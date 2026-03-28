from pathlib import Path


INPUT_DIR = Path("inputs")
TARGET_FILE = INPUT_DIR / "tourism_monthly_dataset.tsf"


def parse_tsf_file(file_path: Path) -> tuple[dict, list[dict]]:
    # .tsf 파일은 헤더(@attribute, @frequency 등)와 실제 데이터(@data 이후)로 나뉩니다.
    # 분석 전에 필요한 정보만 빠르게 보기 위해 헤더와 샘플 레코드만 가볍게 파싱합니다.
    metadata: dict = {
        "relation": None,
        "attributes": [],
        "frequency": None,
        "horizon": None,
        "missing": None,
        "equallength": None,
    }
    records: list[dict] = []
    in_data_section = False

    with file_path.open("r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()

            # 빈 줄과 주석은 건너뜁니다.
            if not line or line.startswith("#"):
                continue

            if line == "@data":
                in_data_section = True
                continue

            if not in_data_section:
                if line.startswith("@relation"):
                    metadata["relation"] = line.split(maxsplit=1)[1]
                elif line.startswith("@attribute"):
                    parts = line.split()
                    metadata["attributes"].append(
                        {"name": parts[1], "type": parts[2]}
                    )
                elif line.startswith("@frequency"):
                    metadata["frequency"] = line.split(maxsplit=1)[1]
                elif line.startswith("@horizon"):
                    metadata["horizon"] = line.split(maxsplit=1)[1]
                elif line.startswith("@missing"):
                    metadata["missing"] = line.split(maxsplit=1)[1]
                elif line.startswith("@equallength"):
                    metadata["equallength"] = line.split(maxsplit=1)[1]
                continue

            # 현재 파일은 "메타값:메타값:시계열값들" 구조를 가집니다.
            parts = line.split(":")
            if len(parts) < 3:
                continue

            series_name = parts[0]
            start_timestamp = parts[1]
            values = [value for value in parts[2].split(",") if value != "?"]

            records.append(
                {
                    "series_name": series_name,
                    "start_timestamp": start_timestamp,
                    "length": len(values),
                    "head_values": values[:5],
                }
            )

    return metadata, records


def summarize_tsf_file(file_path: Path) -> None:
    metadata, records = parse_tsf_file(file_path)

    print(f"\n[FILE] {file_path}")
    print(f"- relation: {metadata['relation']}")
    print(f"- frequency: {metadata['frequency']}")
    print(f"- horizon: {metadata['horizon']}")
    print(f"- missing: {metadata['missing']}")
    print(f"- equallength: {metadata['equallength']}")
    print(f"- attributes: {metadata['attributes']}")
    print(f"- series count: {len(records)}")

    if not records:
        print("- no records found")
        return

    lengths = [record["length"] for record in records]
    print(f"- min length: {min(lengths)}")
    print(f"- max length: {max(lengths)}")
    print(f"- sample series:")

    # 전체를 다 찍지 않고 앞쪽 몇 개만 보여줘서
    # 파일 성격을 빠르게 파악할 수 있게 합니다.
    for record in records[:5]:
        print(
            "  "
            f"{record['series_name']} | "
            f"start={record['start_timestamp']} | "
            f"length={record['length']} | "
            f"head={record['head_values']}"
        )


def main() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"파일이 없습니다: {TARGET_FILE}")

    # 현재는 inputs 폴더 안의 tourism_monthly_dataset.tsf 파일만
    # 빠르게 확인하는 용도로 고정해 둡니다.
    summarize_tsf_file(TARGET_FILE)


if __name__ == "__main__":
    main()
