import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier

def main():
    print("Loan Prediction Pipeline (CatBoost, без event_label)")

    df = pd.read_csv("prikoluha.csv")

    df["target"] = (df["event_action"] == "sub_submit_success").astype(int)

    cat_features = [
        "utm_medium",
        "device_category",
        "device_brand",
        "device_screen_resolution",
        "device_browser",
        "geo_country",
        "geo_city",
        "brand",
        "model"
    ]

    for col in cat_features:
        df[col] = df[col].astype(str).fillna("unknown")

    X = df.drop(
        columns=[
            "event_action", "target", "Unnamed: 0.1", "Unnamed: 0",
            "event_category", "event_category_clean",
            "event_label", "hit_referer", "hit_revent_labeleferer"
        ],
        errors="ignore"
    )

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )



    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        cat_features=cat_features,
        class_weights=[1, 100],
        task_type="GPU",
        devices="0",
        verbose=100,
        early_stopping_rounds=50
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))



    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    # Feature Importance
    fi_values = model.get_feature_importance()
    fi_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": fi_values
    }).sort_values("Importance", ascending=False)

    print(fi_df.head(20))

    joblib.dump(model, "loan_catboost.pkl")
    fi_df.to_csv("feature_importance.csv", index=False)
    print("Модель сохранена в loan_catboost.pkl")
    print("Feature importance сохранены в feature_importance.csv")
    print("Feature names used for training (JSON keys):")
    print(model.feature_names_)


if __name__ == "__main__":
    main()
