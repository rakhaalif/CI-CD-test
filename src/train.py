import json
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train(random_state: int = 42):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=random_state,
        stratify=data.target,
    )
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, {"accuracy": float(acc)}


if __name__ == "__main__":
    model, metrics = train()
    dump(model, "model.joblib")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"accuracy={metrics['accuracy']:.3f}")
