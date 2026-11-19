import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sarcastone.config import *


def create_audio_pipeline(classifier_type='svm'):
    """Create a scikit-learn pipeline for audio classification."""
    if classifier_type == 'svm':
        clf = SVC(C=10.0, gamma='scale', kernel='rbf', probability=True, random_state=RANDOM_STATE)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    elif classifier_type == 'gb':
        clf = GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])


def train_audio_model(train_features, train_labels, classifier_type='svm'):
    """Train the audio sarcasm classifier."""
    pipeline = create_audio_pipeline(classifier_type)
    pipeline.fit(train_features, train_labels)
    return pipeline


def evaluate_audio_model(pipeline, test_features, test_labels):
    """Evaluate the audio model and return metrics."""
    predictions = pipeline.predict(test_features)

    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        probs = pipeline.predict_proba(test_features)
        confidences = np.max(probs, axis=1).tolist()
    else:
        confidences = [1.0] * len(predictions)

    metrics = {
        'accuracy': float(accuracy_score(test_labels, predictions)),
        'f1': float(f1_score(test_labels, predictions, average='weighted')),
        'report': classification_report(test_labels, predictions, output_dict=True, digits=4),
        'confusion_matrix': confusion_matrix(test_labels, predictions).tolist(),
        'predictions': predictions.tolist(),
        'true_labels': test_labels.tolist(),
        'confidences': confidences
    }
    return metrics


def predict_audio(pipeline, features):
    """Predict sarcasm from audio features with confidence."""
    prediction = pipeline.predict(features.reshape(1, -1))[0]
    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        probs = pipeline.predict_proba(features.reshape(1, -1))[0]
        confidence = float(np.max(probs))
    else:
        confidence = 1.0

    return int(prediction), confidence


def save_audio_model(pipeline, path=None):
    """Save trained audio model pipeline."""
    if path is None:
        path = os.path.join(MODELS_DIR, 'audio_model.joblib')
    joblib.dump(pipeline, path)
    print(f"Audio model saved to {path}")


def load_audio_model(path=None):
    """Load trained audio model pipeline."""
    if path is None:
        path = os.path.join(MODELS_DIR, 'audio_model.joblib')
    return joblib.load(path)
