import pandas as pd

df = pd.read_csv("spotify-tracks.dataset.csv").sample(n=5000, random_state=42)
df.to_csv("custom_dataset.csv", index=False)
