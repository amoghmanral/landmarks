import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.1000wondersoftheworld.com/1000-wonders-of-the-world.html"
soup = BeautifulSoup(requests.get(url).content, "html.parser")

cells = []
for td in soup.find_all("td"):
    text = td.get_text(strip=True)
    if text.strip() != "":
        cells.append(text)

rows = []
for i in range(0, len(cells), 4):
    row = cells[i:i+4]
    if len(row) == 4:
        rows.append(row)

df = pd.DataFrame(rows, columns=["Type", "Name", "Country", "Tags"])
df.to_csv("landmark_list.csv", index=False)
print(f"Saved {len(df)} landmarks")