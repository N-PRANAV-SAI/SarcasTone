import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold
from sarcastone.config import *


def load_dataset() -> pd.DataFrame:
    """Load MUStARD dataset into a DataFrame."""
    with open(SARCASM_DATA_PATH) as f:
        data = json.load(f)

    rows = []
    for utterance_id, entry in data.items():
        rows.append({
            'id': utterance_id,
            'utterance': entry['utterance'],
            'speaker': entry['speaker'],
            'context': entry['context'],
            'context_speakers': entry.get('context_speakers', []),
            'show': entry.get('show', 'UNKNOWN'),
            'sarcasm': int(entry['sarcasm'])
        })

    df = pd.DataFrame(rows)
    return df


def load_audio_features() -> Dict[str, np.ndarray]:
    """Load pre-extracted audio features from pickle."""
    with open(AUDIO_FEATURES_PATH, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def get_audio_feature_vectors(df: pd.DataFrame, audio_features: Dict[str, np.ndarray]) -> np.ndarray:
    """Get mean-pooled audio feature vectors aligned with DataFrame rows.

    Each audio feature array has shape (283, T) where T varies.
    We mean-pool across time to get a fixed 283-dimensional vector per utterance.
    """
    vectors = []
    for uid in df['id']:
        if uid in audio_features:
            feat = audio_features[uid]
            vectors.append(np.mean(feat, axis=1))
        else:
            vectors.append(np.zeros(283))
    return np.array(vectors)


def load_split_indices() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load pre-computed 5-fold stratified split indices."""
    with open(SPLIT_INDICES_PATH, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def get_stratified_kfold(labels: np.ndarray, n_splits: int = N_FOLDS) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create new stratified k-fold splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return [(train_idx, test_idx) for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels)]
