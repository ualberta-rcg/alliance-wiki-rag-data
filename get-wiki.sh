#!/bin/bash
BASE_URL="https://docs.alliancecan.ca/w/api.php"
PREFIX="Technical_documentation"
OUTPUT_DIR="alliance_docs"

mkdir -p "$OUTPUT_DIR"

echo "Fetching list of pages under $PREFIX..."

# Step 1: Get all page titles with that prefix
titles=$(curl -s "$BASE_URL?action=query&list=allpages&apprefix=${PREFIX}&format=json&aplimit=max" | jq -r '.query.allpages[].title')

for title in $titles; do
    # URL encode the title
    encoded_title=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$title'''))")
    filename=$(echo "$title" | sed 's/\//_/g' | sed 's/ /_/g').txt

    echo "Downloading: $title"

    # Step 2: Get plain text of each page
    content=$(curl -s "$BASE_URL?action=parse&page=$encoded_title&format=json&prop=text" | jq -r '.parse.text["*"]' | lynx -stdin -dump)

    echo "$content" > "$OUTPUT_DIR/$filename"
done

echo "✅ Done. Saved to ./$OUTPUT_DIR/"

