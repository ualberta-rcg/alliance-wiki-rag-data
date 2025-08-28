import requests
from pathlib import Path
from bs4 import BeautifulSoup

URL = "https://status.alliancecan.ca/"
OUTPUT_DIR = Path("wiki_pages")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "status_page.txt"

def fetch_text():
    resp = requests.get(URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Collapse whitespace into neat lines
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def main():
    text = fetch_text()
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print(f"✅ Saved status page text to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
