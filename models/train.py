import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load configuration files
with open("config/logistic_regression.yaml", "r") as f1:
    logistic_regression_config = yaml.safe_load(f1)

with open("config/random_forest.yaml", "r") as f2:
    random_forest_config = yaml.safe_load(f2)

# 2. Initialize models
lr_model = LogisticRegression(**logistic_regression_config["params"])
rf_model = RandomForestClassifier(**random_forest_config["params"])

# 3. Create a dictionary to hold the models
models = {
    "logistic_regression": lr_model,
    "random_forest": rf_model
}

# 4. Function to retrieve a model by name
def get_model(model_name: str):
    """
    Retrieve a model by its name.
    """
    return models.get(model_name)

def train_test_val_model(model_name: str, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
    """
    Train the specified model on the provided training data.
    
    Args:
        model_name (str): Name of the model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        The trained model.
    """
    model = get_model(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found.")

    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        val_predictions = model.predict(X_val)
        print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions)}")
        print(f"Validation Precision: {precision_score(y_val, val_predictions)}")
        print(f"Validation Recall: {recall_score(y_val, val_predictions)}")
        print(f"Validation F1 Score: {f1_score(y_val, val_predictions)}")

    if X_test is not None and y_test is not None:
        test_predictions = model.predict(X_test)
        print(f"Test Accuracy: {accuracy_score(y_test, test_predictions)}")
        print(f"Test Precision: {precision_score(y_test, test_predictions)}")
        print(f"Test Recall: {recall_score(y_test, test_predictions)}")
        print(f"Test F1 Score: {f1_score(y_test, test_predictions)}")

    return model

if __name__ == "__main__":
    # Example usage
    model_name = "logistic_regression"
    model = get_model(model_name)
    print(f"Retrieved model: {model}")