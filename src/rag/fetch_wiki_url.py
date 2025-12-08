import pandas as pd
import requests
import time

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "project/1.0"}

def get_wiki_url(title, session):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "info",
        "inprop": "url",
        "redirects": True,
    }
    data = session.get(WIKI_API, params=params, headers=HEADERS).json()
    
    for page_id, page in data["query"]["pages"].items():
        if page_id != "-1":
            return page.get("fullurl")
    return None


def search_wiki(query, session):
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
    }
    data = session.get(WIKI_API, params=params, headers=HEADERS).json()
     
    results = data["query"]["search"]
    if results:
        return results[0]["title"]
    return None


def main():
    df = pd.read_csv("../../src/data/train.csv")
    landmarks = df['landmark_name'].unique()
    print(f"Fetching {len(landmarks)} landmarks\n")
    
    session = requests.Session()
    results = []
    
    for i, name in enumerate(landmarks):
        print(f"[{i+1}/{len(landmarks)}] {name}...", end=" ")
        
        url = get_wiki_url(name, session)
        method = "direct"
        
        if not url:
            title = search_wiki(name, session)
            if title:
                url = get_wiki_url(title, session)
                method = "search"
        
        if url:
            results.append({"landmark_name": name, "wiki_url": url, "fetch_method": method})
        else:
            print(f"Failed: {name}")
        
        time.sleep(0.1)
    
    pd.DataFrame(results).to_csv("wiki-context.csv", index=False)
    print(f"Saved {len(results)} landmarks")


if __name__ == "__main__":
    main()