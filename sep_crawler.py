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