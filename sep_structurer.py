import os
import re
import csv


TEXT_DIR = "data/sep_text"
OUTPUT_PATH = "sep_values.csv"


def extract_date_from_filename(filename):
    match = re.search(r"fomcprojtabl(\d{8})", filename)
    if not match:
        return None

    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def normalize_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def find_values(text, label, count):
    pattern = rf"{re.escape(label)}\s+([-]?\d+(?:\.\d+)?(?:–[-]?\d+(?:\.\d+)?)?(?:\s+[-]?\d+(?:\.\d+)?(?:–[-]?\d+(?:\.\d+)?)?){{{count - 1}}})"
    match = re.search(pattern, text)

    if not match:
        return [None] * count

    values = match.group(1).split()
    values = values[:count]

    if len(values) < count:
        values += [None] * (count - len(values))

    return values


def parse_sep_text(text):
    text = normalize_text(text)

    gdp = find_values(text, "Change in real GDP", 4)
    unrate = find_values(text, "Unemployment rate", 4)
    pce = find_values(text, "PCE inflation", 4)
    core_pce = find_values(text, "Core PCE inflation4", 3)

    if core_pce == [None, None, None]:
        core_pce = find_values(text, "Core PCE inflation", 3)

    fedfunds = find_values(text, "Federal funds rate", 4)

    return {
        "GDP_Year1": gdp[0],
        "GDP_Year2": gdp[1],
        "GDP_Year3": gdp[2],
        "GDP_LongerRun": gdp[3],
        "UNRATE_Year1": unrate[0],
        "UNRATE_Year2": unrate[1],
        "UNRATE_Year3": unrate[2],
        "UNRATE_LongerRun": unrate[3],
        "PCE_Year1": pce[0],
        "PCE_Year2": pce[1],
        "PCE_Year3": pce[2],
        "PCE_LongerRun": pce[3],
        "CORE_PCE_Year1": core_pce[0],
        "CORE_PCE_Year2": core_pce[1],
        "CORE_PCE_Year3": core_pce[2],
        "FEDFUNDS_Year1": fedfunds[0],
        "FEDFUNDS_Year2": fedfunds[1],
        "FEDFUNDS_Year3": fedfunds[2],
        "FEDFUNDS_LongerRun": fedfunds[3],
    }


def build_sep_rows():
    rows = []

    if not os.path.exists(TEXT_DIR):
        print(f"[ERROR] Missing folder: {TEXT_DIR}")
        return rows

    for filename in sorted(os.listdir(TEXT_DIR)):
        if not filename.endswith(".txt"):
            continue

        date = extract_date_from_filename(filename)
        if not date:
            print(f"[SKIP] {filename}")
            continue

        path = os.path.join(TEXT_DIR, filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            values = parse_sep_text(text)
            row = {"Date": date, **values}
            rows.append(row)
            print(f"[DONE] {filename}")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    return rows


def save_sep_csv(rows):
    fieldnames = [
        "Date",
        "GDP_Year1",
        "GDP_Year2",
        "GDP_Year3",
        "GDP_LongerRun",
        "UNRATE_Year1",
        "UNRATE_Year2",
        "UNRATE_Year3",
        "UNRATE_LongerRun",
        "PCE_Year1",
        "PCE_Year2",
        "PCE_Year3",
        "PCE_LongerRun",
        "CORE_PCE_Year1",
        "CORE_PCE_Year2",
        "CORE_PCE_Year3",
        "FEDFUNDS_Year1",
        "FEDFUNDS_Year2",
        "FEDFUNDS_Year3",
        "FEDFUNDS_LongerRun",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVE] {OUTPUT_PATH}")


def main():
    rows = build_sep_rows()
    save_sep_csv(rows)


if __name__ == "__main__":
    main()