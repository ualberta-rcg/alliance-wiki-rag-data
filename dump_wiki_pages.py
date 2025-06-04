import requests
import os

API_ENDPOINT = "https://docs.alliancecan.ca/mediawiki/api.php"
OUTPUT_DIR = "wiki_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_pages():
    pages = []
    params = {
        "action": "query",
        "list": "allpages",
        "format": "json",
        "aplimit": "max"
    }
    while True:
        res = requests.get(API_ENDPOINT, params=params).json()
        pages += res["query"]["allpages"]
        if "continue" in res:
            params.update(res["continue"])
        else:
            break
    return pages

def get_page_content(title):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "titles": title
    }
    res = requests.get(API_ENDPOINT, params=params).json()
    pages = res.get("query", {}).get("pages", {})
    for page_id, data in pages.items():
        if "revisions" in data:
            return data["revisions"][0]["*"] if "*" in data["revisions"][0] else data["revisions"][0]["slots"]["main"]["*"]
    return ""

def save_page(title, content):
    safe_title = title.replace("/", "_")
    with open(os.path.join(OUTPUT_DIR, f"{safe_title}.txt"), "w") as f:
        f.write(content)

def main():
    print("🔍 Fetching page list...")
    pages = get_all_pages()
    print(f"📄 Found {len(pages)} pages. Downloading...")
    for page in pages:
        title = page["title"]
        print(f"→ {title}")
        content = get_page_content(title)
        save_page(title, content)
    print(f"✅ Done. Saved to ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

