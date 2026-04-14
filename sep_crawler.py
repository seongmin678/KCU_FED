import os
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
SAVE_DIR = "data/sep"


def get_sep_links():
    response = requests.get(CALENDAR_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"]

        if "fomcprojtabl" in href:
            full_url = urljoin(BASE_URL, href)
            if full_url.lower().endswith(".pdf"):
                links.add(full_url)

    return sorted(links)


def download_sep_pdfs():
    os.makedirs(SAVE_DIR, exist_ok=True)

    pdf_links = get_sep_links()

    for pdf_url in pdf_links:
        filename = os.path.basename(pdf_url)
        filepath = os.path.join(SAVE_DIR, filename)

        if os.path.exists(filepath):
            print(f"[SKIP] {filename}")
            continue

        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"[DONE] {filename}")
        time.sleep(1)


if __name__ == "__main__":
    download_sep_pdfs()

import os
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
PDF_DIR = "data/sep"
TEXT_DIR = "data/sep_text"


def get_sep_links():
    response = requests.get(CALENDAR_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"]

        if "fomcprojtabl" in href:
            full_url = urljoin(BASE_URL, href)
            if full_url.lower().endswith(".pdf"):
                links.add(full_url)

    return sorted(links)


def download_sep_pdfs():
    os.makedirs(PDF_DIR, exist_ok=True)

    pdf_links = get_sep_links()
    downloaded_files = []

    for pdf_url in pdf_links:
        filename = os.path.basename(pdf_url)
        filepath = os.path.join(PDF_DIR, filename)

        if os.path.exists(filepath):
            print(f"[SKIP] {filename}")
            downloaded_files.append(filepath)
            continue

        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"[DONE] {filename}")
        downloaded_files.append(filepath)
        time.sleep(1)

    return downloaded_files


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    texts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)

    return "\n\n".join(texts)


def save_sep_texts(pdf_files):
    os.makedirs(TEXT_DIR, exist_ok=True)

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        text_filename = filename.replace(".pdf", ".txt")
        text_path = os.path.join(TEXT_DIR, text_filename)

        if os.path.exists(text_path):
            print(f"[SKIP TEXT] {text_filename}")
            continue

        try:
            text = extract_text_from_pdf(pdf_path)

            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"[TEXT] {text_filename}")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")


def main():
    pdf_files = download_sep_pdfs()
    save_sep_texts(pdf_files)


if __name__ == "__main__":
    main()


import os
import re
import csv


TEXT_DIR = "data/sep_text"
OUTPUT_PATH = "data/sep_values.csv"


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
    os.makedirs("data", exist_ok=True)

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