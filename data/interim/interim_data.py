import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import load_data_config, load_paths  # noqa: E402


def shannon_entropy(s):
    """Calculate the Shannon entropy of a string."""
    if not s:
        return 0
    counts = Counter(s)
    entropy_value = entropy(list(counts.values()), base=2)
    return entropy_value

def clean_raw_data(raw_data_path: Path, interim_path: Path, drop_columns: list[str]) -> None:
    """
    Cleans the raw data and saves it to the interim data path.

    Args:
        raw_data_path (Path): Path to the raw data file.
        interim_path (Path): Path to save the cleaned interim data.
        drop_columns (list[str]): Raw columns to discard once feature engineering is done.
    """
    print("Cleaning raw data...")

    # Load the raw data
    raw_data = pd.read_csv(raw_data_path)

    # Drop missing values. Columns in drop_columns are kept for now since
    # Domain is still needed below for feature engineering.
    cleaned_data = raw_data.dropna()

    print("Feature engineering...")

    # Count hyphens in the domain
    cleaned_data['n_hyphens_in_domain'] = cleaned_data['Domain'].apply(lambda x: str(x).count('-')) 

    # Calculate the ratio of hyphens to domain length
    cleaned_data['hyphen_to_domain_length_ratio'] = cleaned_data['n_hyphens_in_domain'] / cleaned_data['DomainLength']

    # Calculate the ratio of domain length to URL length
    cleaned_data['domainToURLLengthRatio'] = cleaned_data['DomainLength'] / cleaned_data['URLLength']

    # Calculate the Shannon entropy of the domain
    cleaned_data['domain_entropy'] = cleaned_data['Domain'].apply(lambda x: shannon_entropy(str(x)))

    # Check if the domain is internationalized (starts with 'xn--')
    cleaned_data['is_international_domain'] = cleaned_data['Domain'].apply(lambda x: 1 if str(x).startswith('xn--') else 0)

    # Now that Domain-derived features are computed, drop the raw columns.
    cleaned_data = cleaned_data.drop(columns=drop_columns, errors='ignore')

    # Save the cleaned data to the interim path
    interim_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    cleaned_data.to_csv(interim_path, index=False)

    print(f"Cleaned data saved to {interim_path}")

if __name__ == "__main__":
    paths = load_paths()
    data_config = load_data_config()

    clean_raw_data(
        paths["raw_data"],
        paths["interim_data"],
        drop_columns=data_config["drop_columns"],
    )