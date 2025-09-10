import pandas as pd
from sklearn.datasets import make_classification

def generate_and_save_data(n_samples=1000, n_features=10, n_classes=2, random_state=42, path='data/synthetic_data.csv'):
    """
    Generates synthetic classification data and saves it to a CSV file.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=5, n_redundant=2,
                               n_classes=n_classes, random_state=random_state)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    df.to_csv(path, index=False)
    print(f"Data generated and saved to {path}")

if __name__ == "__main__":
    generate_and_save_data()
