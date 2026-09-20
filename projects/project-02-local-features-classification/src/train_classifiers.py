
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import BayesianGaussianMixture

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    r"C:\Users\asusf\OneDrive\Documentos\Doctorado\semestre 4\clasficadores RMIB\BDcric30images.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "cric"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_FRACTION = 1.0
RANDOM_SEED = 42


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(data_path, fraction=1.0, random_seed=42):
    """Load the dataset and separate features from labels."""

    start_time = time()

    print("\nLoading dataset...")
    data = pd.read_csv(data_path)

    print(f"Original shape: {data.shape}")

    if fraction < 1.0:
        data = (
            data.groupby(
                data.columns[-1],
                group_keys=False
            )
            .apply(
                lambda group: group.sample(
                    frac=fraction,
                    random_state=random_seed
                )
            )
            .reset_index(drop=True)
        )

    X = data.iloc[:, :-1].astype(np.float32).values
    y = data.iloc[:, -1].astype(np.int32).values

    print(f"Selected shape: {data.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Loading time: {time() - start_time:.2f} seconds")

    return X, y


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate binary classification metrics."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0 else 0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# CLASSIFIER FACTORIES
# ============================================================

def create_standard_classifiers():
    """
    Create all standard classifiers.

    SVM is intentionally excluded because of its
    expected computational cost on this dataset.
    """

    return {
        "knn": KNeighborsClassifier(
            n_neighbors=4,
            n_jobs=-1
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_SEED
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=50,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }


# ============================================================
# STANDARD CLASSIFIERS
# ============================================================

def train_standard_classifier(
    classifier_name,
    classifier,
    X_train,
    X_test,
    y_train,
    y_test
):
    """Train and evaluate a standard classifier."""

    print("\n" + "=" * 60)
    print(f"CLASSIFIER: {classifier_name.upper()}")
    print("=" * 60)

    training_start = time()

    classifier.fit(X_train, y_train)

    training_time = time() - training_start

    print(f"Training time: {training_time:.2f} seconds")

    prediction_start = time()

    y_pred = classifier.predict(X_test)

    prediction_time = time() - prediction_start

    print(f"Prediction time: {prediction_time:.2f} seconds")

    metrics = calculate_metrics(y_test, y_pred)

    metrics["classifier"] = classifier_name
    metrics["training_time_seconds"] = training_time
    metrics["prediction_time_seconds"] = prediction_time

    return metrics


# ============================================================
# BAYESIAN GAUSSIAN MIXTURE
# ============================================================

def train_bayesian_gmm(X_train, X_test, y_train, y_test):
    """
    Train Bayesian Gaussian Mixture with four components.

    The unsupervised components are mapped to classes
    using the majority class in the training data.
    """

    print("\n" + "=" * 60)
    print("CLASSIFIER: BAYESIAN GMM")
    print("=" * 60)

    training_start = time()

    model = BayesianGaussianMixture(
        n_components=4,
        covariance_type="full",
        random_state=RANDOM_SEED,
        max_iter=100
    )

    model.fit(X_train)

    training_time = time() - training_start

    print(f"Training time: {training_time:.2f} seconds")

    # Map each cluster to its majority training class
    train_clusters = model.predict(X_train)

    cluster_to_class = {}

    for cluster_id in range(model.n_components):

        cluster_labels = y_train[
            train_clusters == cluster_id
        ]

        if len(cluster_labels) == 0:
            cluster_to_class[cluster_id] = 0
        else:
            cluster_to_class[cluster_id] = np.bincount(
                cluster_labels
            ).argmax()

    prediction_start = time()

    test_clusters = model.predict(X_test)

    y_pred = np.array([
        cluster_to_class.get(cluster_id, 0)
        for cluster_id in test_clusters
    ])

    prediction_time = time() - prediction_start

    print(f"Prediction time: {prediction_time:.2f} seconds")

    metrics = calculate_metrics(y_test, y_pred)

    metrics["classifier"] = "bayesian_gmm"
    metrics["training_time_seconds"] = training_time
    metrics["prediction_time_seconds"] = prediction_time

    return metrics


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    total_start = time()

    print("=" * 60)
    print("CRIC CLASSIFICATION EXPERIMENT")
    print("All classifiers")
    print("=" * 60)

    # Load the complete dataset
    X, y = load_dataset(
        DATA_PATH,
        fraction=DATA_FRACTION,
        random_seed=RANDOM_SEED
    )

    # Use the same stratified split for every classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED
    )

    print(f"\nTraining samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")

    all_results = []

    # Train KNN, Decision Tree, and Random Forest
    classifiers = create_standard_classifiers()

    for classifier_name, classifier in classifiers.items():

        try:

            metrics = train_standard_classifier(
                classifier_name,
                classifier,
                X_train,
                X_test,
                y_train,
                y_test
            )

            all_results.append(metrics)

            print("\nMetrics:")
            for key, value in metrics.items():
                print(f"{key}: {value}")

        except Exception as error:

            print(
                f"\nError in {classifier_name}: {error}"
            )

    # Train Bayesian Gaussian Mixture
    try:

        metrics = train_bayesian_gmm(
            X_train,
            X_test,
            y_train,
            y_test
        )

        all_results.append(metrics)

        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    except Exception as error:

        print(f"\nError in Bayesian GMM: {error}")

    # Save all results
    total_time = time() - total_start

    results_df = pd.DataFrame(all_results)

    results_df["data_fraction"] = DATA_FRACTION
    results_df["total_experiment_time_seconds"] = total_time

    results_file = RESULTS_DIR / "cric_all_classifiers_metrics.csv"

    results_df.to_csv(
        results_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print(f"\nTotal experiment time: {total_time:.2f} seconds")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()