import pandas as pd

file_path = "training.1600000.processed.noemoticon.csv"
cols = ["target", "ids", "date", "flag", "user", "text"]

df = pd.read_csv(file_path, encoding="latin-1", names=cols)
# count the # of labeled pos/neg/neutral tweets
positive_count = (df["target"] == 4).sum()
negative_count = (df["target"] == 0).sum()
neutral_count = (df["target"] == 2).sum()
ratio = positive_count / negative_count

print(f"Positive tweets: {positive_count}")
print(f"Negative tweets: {negative_count}")
print(f"Netural tweets: {neutral_count}")
print(f"Ratio (positive : negative) = {ratio:.2f}")
