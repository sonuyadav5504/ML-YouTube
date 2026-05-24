import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from models import (
    GaussianNaiveBayesScratch,
    LDAScratch,
    QDAScratch
)


def load_wine_dataset():
    data = load_wine()

    X = data.data
    y = data.target

    df = pd.DataFrame(X, columns=data.feature_names)
    df["target"] = y
    df["target_name"] = df["target"].apply(lambda i: data.target_names[i])

    return data, X, y, df


def train_all_models(test_size=0.2, random_state=42):
    data, X, y, df = load_wine_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Naive Bayes From Scratch": GaussianNaiveBayesScratch(),
        "GDA / LDA From Scratch": LDAScratch(),
        "QDA From Scratch": QDAScratch(),
        "Sklearn GaussianNB": GaussianNB(),
        "Sklearn LDA": LinearDiscriminantAnalysis(),
        "Sklearn QDA": QuadraticDiscriminantAnalysis()
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy
        })

        trained_models[model_name] = {
            "model": model,
            "y_pred": y_pred,
            "y_test": y_test
        }

    results_df = pd.DataFrame(results).sort_values(
        by="Accuracy",
        ascending=False
    )

    return {
        "data": data,
        "df": df,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "results_df": results_df,
        "trained_models": trained_models
    }


if __name__ == "__main__":
    output = train_all_models()

    print("\nModel Accuracy Comparison:")
    print(output["results_df"])