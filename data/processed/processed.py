import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import load_data_config, load_paths  # noqa: E402


def process_data(
    interim_data_path: Path,
    processed_dir: Path,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
):
    """
    Process the interim data for modeling.

    Args:
        interim_data_path (Path): Path to the interim data file.
        processed_dir (Path): Directory to write the split CSVs to.
        target_column (str): Name of the label column.
        test_size (float): Fraction of the full dataset held out as the test set.
        val_size (float): Fraction of the train+val split held out as the validation set.
        random_state (int): Seed used for both splits.

    Returns:
        None
    """

    print("Processing interim data...")

    # Load the interim data
    df = pd.read_csv(interim_data_path)

    # Split the data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 1: Split the data into training + validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 2: Split the training + validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    # With test_size=0.1 and val_size=0.1111, this yields ~80% train, 10% test, 10% val.

    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    X_val.to_csv(processed_dir / "X_val.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)
    y_val.to_csv(processed_dir / "y_val.csv", index=False)

    print(f"Processed data saved to {processed_dir}")

if __name__ == "__main__":
    paths = load_paths()
    data_config = load_data_config()

    process_data(
        paths["interim_data"],
        paths["processed_dir"],
        target_column=data_config["target_column"],
        test_size=data_config["split"]["test_size"],
        val_size=data_config["split"]["val_size"],
        random_state=data_config["split"]["random_state"],
    )
