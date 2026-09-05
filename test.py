import pandas as pd
df = pd.read_csv("priority_auto_labelled.csv")
print("Finalized:", (df["priority"] != "").sum(), "/", len(df))
print(df[df["priority"] != ""]["priority"].value_counts())
print("\nZero-shot predicted (unfinalized) distribution:")
print(df["predicted_priority"].value_counts())