import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/csv/features.csv"


def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist yet. Generate check-ins first!")
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded : {len(df)} records (checkin_records) found.\n")
    return df


def display_stats(df: pd.DataFrame):
    print("=== DESCRIPTIVE STATISTICS ===")
    numeric_cols = [col for col in ['average_eyes_ear','mouth_openness_mar', 'smile_score', 'mood'] if col in df.columns]
    print(df[numeric_cols].describe().round(3))
    print("\n")
    print(df["mood"].value_counts().sort_index())
    print("\n")


def compute_correlations(df: pd.DataFrame):
    print("=== CORRELATION MATRIX (Pearson) ===")
    numeric_cols = [col for col in ['average_eyes_ear','mouth_openness_mar', 'smile_score', 'mood'] if col in df.columns]
    corr_matrix = df[numeric_cols].corr().round(2)
    print(corr_matrix)
    return corr_matrix

def display_histograms(df: pd.DataFrame):
   for column in ["average_eyes_ear","mouth_openness_mar","smile_score"]:
    plt.figure()

    plt.hist(df[column], bins=10)

    plt.title(column)

    plt.show()

if __name__ == "__main__":
    dataset = load_dataset(CSV_PATH)
    display_stats(dataset)
    correlations = compute_correlations(dataset)
    display_histograms(dataset)