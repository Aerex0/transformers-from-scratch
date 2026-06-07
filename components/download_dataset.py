import kagglehub
from pathlib import Path

# Download the WMT 2014 English-German dataset using KaggleHub
path = kagglehub.dataset_download(
    "mohamedlotfy50/wmt-2014-english-german"
)

print("Downloaded to:", path)

for p in Path(path).rglob("*"):
    print(p)