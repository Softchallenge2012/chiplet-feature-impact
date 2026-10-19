import pandas as pd

df = pd.read_csv("data/synthetic_similarity_labeled_1000.csv")

print(f"predicted_label==0shape: {df[df['predicted_label']==0].shape}")