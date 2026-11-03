"""
prediction.py

Run graph-based rule predictions against the SECOM dataset and print
evaluation metrics (confusion matrix, classification report, main metrics).

Usage:
    python .\prediction.py
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
import warnings

# Configuration
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "12345678")
DATA_PATH = r"Dataset\secom.data"
LABELS_PATH = r"Dataset\secom_labels.data"

warnings.filterwarnings("ignore")


class PredictiveMaintenanceEvaluator:
    """
    Fetches rules from Neo4j and evaluates a batch of feature rows
    against those rules.

    Methods:
        batch_predict(features_df) -> (predictions, probabilities)
    """

    def __init__(self, uri: str, auth: tuple):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.driver.verify_connectivity()

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def batch_predict(self, features_df: pd.DataFrame):
        """
        Run rule checks for each row in `features_df`.

        Returns:
            predictions: list[int] (0 or 1)
            probabilities: list[float] (confidence scores)
        """
        predictions = []
        probabilities = []

        active_rules = self._fetch_all_rules()
        print(f"   -> Loaded {len(active_rules)} active rules from Knowledge Graph.")

        total = len(features_df)
        print(f"   -> Evaluating {total} test points...")

        for idx, row in features_df.iterrows():
            is_failure = False
            max_confidence = 0.0

            for rule in active_rules:
                sensor_val = row.get(rule["sensor"], 0)

                threshold_met = (
                    (rule["condition"] == "GT" and sensor_val > rule["threshold"])
                    or (rule["condition"] == "LT" and sensor_val < rule["threshold"])
                )

                if threshold_met:
                    is_failure = True
                    if rule["confidence"] > max_confidence:
                        max_confidence = rule["confidence"]

            predictions.append(1 if is_failure else 0)
            probabilities.append(max_confidence if is_failure else 0.0)

            if (idx + 1) % 100 == 0:
                print(f"      Processed {idx + 1}/{total}...", end="\r")

        return predictions, probabilities

    def _fetch_all_rules(self):
        """Return list of rule dicts from Neo4j."""
        query = """
        MATCH (r:Rule)
        RETURN r.sensor as sensor,
               r.condition as condition,
               r.threshold as threshold,
               r.precision as confidence
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]


def run_evaluation():
    """Load data, run predictions, and print evaluation metrics."""
    print(">>> 1. LOADING DATA AND GROUND TRUTH...")

    df_x = pd.read_csv(DATA_PATH, sep=" ", header=None)
    df_x.columns = [f"f_{i}" for i in range(df_x.shape[1])]
    df_x = df_x.fillna(0)

    df_y = pd.read_csv(LABELS_PATH, sep=" ", header=None)
    df_y.columns = ["Label", "Timestamp"]
    y = df_y["Label"].apply(lambda x: 1 if x == 1 else 0)

    print(f"    Loaded {len(df_x)} rows and {df_x.shape[1]} features")
    print(f"    Class distribution: {y.value_counts().to_dict()}")

    # Split data: 80% train, 20% test (same as baseline-xgb)
    X_train, X_test, y_train, y_test = train_test_split(
        df_x, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"    Test set size: {len(X_test)} rows")

    evaluator = PredictiveMaintenanceEvaluator(NEO4J_URI, NEO4J_AUTH)
    try:
        y_pred, y_scores = evaluator.batch_predict(X_test)

        print("\n\n>>> 2. CALCULATION METRICS")
        print("=" * 60)

        # Confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            print("Confusion Matrix:")
            print(f" [ TN: {tn} | FP: {fp} ]")
            print(f" [ FN: {fn} | TP: {tp} ]")
        except ValueError:
            # In case of single-class predictions/labels
            print("Confusion Matrix (raw):")
            print(confusion_matrix(y_test, y_pred))

        print("-" * 30)
        print(classification_report(y_test, y_pred, digits=4))

        # Core metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_scores)
        except ValueError:
            auc = 0.0

        print(f"ACCURACY  : {acc:.4f}")
        print(f"PRECISION : {prec:.4f}")
        print(f"RECALL    : {rec:.4f}")
        print(f"F1-SCORE  : {f1:.4f}")
        print(f"ROC AUC   : {auc:.4f}")
        print("=" * 60)

    finally:
        evaluator.close()


if __name__ == "__main__":
    run_evaluation()