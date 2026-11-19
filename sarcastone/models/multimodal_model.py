import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sarcastone.config import *


def fuse_features(text_embeddings, audio_features):
    """Concatenate text and audio features (early fusion).

    Args:
        text_embeddings: np.ndarray of shape (N, 768) - DistilBERT [CLS] embeddings
        audio_features: np.ndarray of shape (N, 283) - mean-pooled audio features

    Returns:
        np.ndarray of shape (N, 1051) - fused features
    """
    text_embeddings = np.nan_to_num(text_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    audio_features = np.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)

    return np.concatenate([text_embeddings, audio_features], axis=1)


def create_multimodal_pipeline(classifier_type='mlp'):
    """Create classification pipeline for fused features."""
    if classifier_type == 'mlp':
        clf = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1
        )
    elif classifier_type == 'svm':
        clf = SVC(C=10.0, gamma='scale', kernel='rbf', probability=True, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])


def train_multimodal_model(train_text_emb, train_audio_feat, train_labels, classifier_type='mlp'):
    """Train the multimodal sarcasm classifier."""
    fused = fuse_features(train_text_emb, train_audio_feat)
    pipeline = create_multimodal_pipeline(classifier_type)
    pipeline.fit(fused, train_labels)
    return pipeline


def evaluate_multimodal_model(pipeline, test_text_emb, test_audio_feat, test_labels):
    """Evaluate the multimodal model and return metrics."""
    fused = fuse_features(test_text_emb, test_audio_feat)
    predictions = pipeline.predict(fused)

    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        probs = pipeline.predict_proba(fused)
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


def predict_multimodal(pipeline, text_embedding, audio_features):
    """Predict sarcasm from fused features with confidence."""
    fused = fuse_features(
        text_embedding.reshape(1, -1),
        audio_features.reshape(1, -1)
    )
    prediction = pipeline.predict(fused)[0]

    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        probs = pipeline.predict_proba(fused)[0]
        confidence = float(np.max(probs))
    else:
        confidence = 1.0

    return int(prediction), confidence


def save_multimodal_model(pipeline, path=None):
    """Save trained multimodal model."""
    if path is None:
        path = os.path.join(MODELS_DIR, 'multimodal_model.joblib')
    joblib.dump(pipeline, path)
    print(f"Multimodal model saved to {path}")


def load_multimodal_model(path=None):
    """Load trained multimodal model."""
    if path is None:
        path = os.path.join(MODELS_DIR, 'multimodal_model.joblib')
    return joblib.load(path)
