import pandas as pd
import requests
import time
from urllib.parse import urlparse, unquote

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "project/1.0"}


def get_title_from_url(url):
    path = urlparse(url).path
    if path.startswith("/wiki/"):
        return unquote(path[6:])
    return None


def get_content(title, session):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "redirects": True,
    }
    data = session.get(WIKI_API, params=params, headers=HEADERS).json()
    
    for page_id, page in data["query"]["pages"].items():
        if page_id != "-1":
            return page.get("extract", "")
    return ""


def main():
    df = pd.read_csv("wiki-context.csv.csv")
    
    session = requests.Session()
    
    for i, row in df.iterrows():        
        title = get_title_from_url(row['wiki_url'])
        content = get_content(title, session) if title else ""
        
        df.at[i, 'page_content'] = content
        df.at[i, 'content_length'] = len(content)
        
        time.sleep(0.1)
    
    df.to_csv("wiki_context.csv", index=False)


if __name__ == "__main__":
    main()