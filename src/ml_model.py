"""
Machine learning model for collision risk prediction.

This model uses physical features derived from satellite motion:
- distance between satellites
- relative velocity
- altitude difference

Goal:
Predict whether a conjunction may lead to a collision.
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def train_model(df):

    X = df[["distance", "relative_velocity"]]
    y = df["collision"]

    # sécurité globale
    if y.nunique() < 2:
        raise ValueError("Dataset has only one class")

    # split 
    if y.value_counts().min() < 2:
        # fallback → no stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

    # sécurité après split
    if y_train.nunique() < 2:
        raise ValueError("Training set has only one class")

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    return model, score
    
    
def predict_risk(model, df):
    """
    Return collision probability.
    """
    X = df[["distance", "relative_velocity"]]
    return model.predict_proba(X)[:, 1]
