import requests
from pathlib import Path
from bs4 import BeautifulSoup
import json

URL = "https://explora.alliancecan.ca/events"
OUTPUT_DIR = Path("wiki_pages")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "events.json"

def fetch_events():
    resp = requests.get(URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    events = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "Event":
                events.append(data)
        except Exception:
            continue
    return events

def main():
    events = fetch_events()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(events)} events to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
