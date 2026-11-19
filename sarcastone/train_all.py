#!/usr/bin/env python3
"""
SarcasTone - Train All Models
Trains text-only, audio-only, and multimodal sarcasm detection models
with stratified 5-fold cross-validation.
"""

import os
import sys
import json
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sarcastone.config import *
from sarcastone.data_utils import (load_dataset, load_audio_features,
                                    get_audio_feature_vectors, get_stratified_kfold)
from sarcastone.models.text_model import (train_text_model, evaluate_text_model,
                                           save_text_model, get_bert_embeddings,
                                           TextSarcasmClassifier)
from sarcastone.models.audio_model import (train_audio_model, evaluate_audio_model,
                                            save_audio_model)
from sarcastone.models.multimodal_model import (train_multimodal_model, evaluate_multimodal_model,
                                                 save_multimodal_model)
from sarcastone.evaluate import compute_metrics, aggregate_cv_results, save_results, print_model_comparison
from transformers import DistilBertTokenizer, DistilBertModel


def train_and_evaluate_all():
    """Train and evaluate all three models."""
    print("=" * 60)
    print("SarcasTone - Training All Models")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    df = load_dataset()
    print(f"  Total samples: {len(df)}")
    print(f"  Sarcastic: {df['sarcasm'].sum()}, Non-sarcastic: {(1 - df['sarcasm']).sum()}")

    audio_features_dict = load_audio_features()
    audio_features = get_audio_feature_vectors(df, audio_features_dict)
    print(f"  Audio features shape: {audio_features.shape}")

    texts = df['utterance'].values
    labels = df['sarcasm'].values

    # Create CV folds
    folds = get_stratified_kfold(labels)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  Device: {device}")

    # ============ TEXT MODEL ============
    print("\n" + "=" * 60)
    print("Training Text-Only Model (DistilBERT)")
    print("=" * 60)

    text_fold_metrics = []
    best_text_model = None
    best_text_tokenizer = None
    best_text_f1 = 0

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        train_texts = texts[train_idx]
        train_labels = labels[train_idx]
        test_texts = texts[test_idx]
        test_labels = labels[test_idx]

        model, tokenizer = train_text_model(
            train_texts, train_labels,
            epochs=TEXT_EPOCHS, device=device
        )

        metrics = evaluate_text_model(model, tokenizer, test_texts, test_labels, device=device)
        text_fold_metrics.append(metrics)
        print(f"  Fold {fold_idx + 1} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        if metrics['f1'] > best_text_f1:
            best_text_f1 = metrics['f1']
            best_text_model = model
            best_text_tokenizer = tokenizer

    text_results = aggregate_cv_results(text_fold_metrics)
    save_results(text_results, 'text_model')
    if best_text_model:
        save_text_model(best_text_model, best_text_tokenizer)

    # ============ AUDIO MODEL ============
    print("\n" + "=" * 60)
    print("Training Audio-Only Model (SVM)")
    print("=" * 60)

    audio_fold_metrics = []
    best_audio_model = None
    best_audio_f1 = 0

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        train_features = audio_features[train_idx]
        train_labels_fold = labels[train_idx]
        test_features = audio_features[test_idx]
        test_labels_fold = labels[test_idx]

        pipeline = train_audio_model(train_features, train_labels_fold)
        metrics = evaluate_audio_model(pipeline, test_features, test_labels_fold)
        audio_fold_metrics.append(metrics)
        print(f"  Fold {fold_idx + 1} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        if metrics['f1'] > best_audio_f1:
            best_audio_f1 = metrics['f1']
            best_audio_model = pipeline

    audio_results = aggregate_cv_results(audio_fold_metrics)
    save_results(audio_results, 'audio_model')
    if best_audio_model:
        save_audio_model(best_audio_model)

    # ============ MULTIMODAL MODEL ============
    print("\n" + "=" * 60)
    print("Training Multimodal Model (Text + Audio Fusion)")
    print("=" * 60)

    # Extract BERT embeddings for all texts
    print("  Extracting DistilBERT embeddings for multimodal fusion...")
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    bert_model = DistilBertModel.from_pretrained(TEXT_MODEL_NAME).to(device)
    bert_model.eval()

    all_text_embeddings = []
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                str(text), add_special_tokens=True, max_length=TEXT_MAX_LENGTH,
                padding='max_length', truncation=True, return_tensors='pt'
            ).to(device)
            outputs = bert_model(**encoding)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
            all_text_embeddings.append(cls_emb)
    all_text_embeddings = np.array(all_text_embeddings)
    print(f"  Text embeddings shape: {all_text_embeddings.shape}")

    multimodal_fold_metrics = []
    best_mm_model = None
    best_mm_f1 = 0

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        train_text_emb = all_text_embeddings[train_idx]
        train_audio_feat = audio_features[train_idx]
        train_labels_fold = labels[train_idx]
        test_text_emb = all_text_embeddings[test_idx]
        test_audio_feat = audio_features[test_idx]
        test_labels_fold = labels[test_idx]

        pipeline = train_multimodal_model(train_text_emb, train_audio_feat, train_labels_fold)
        metrics = evaluate_multimodal_model(pipeline, test_text_emb, test_audio_feat, test_labels_fold)
        multimodal_fold_metrics.append(metrics)
        print(f"  Fold {fold_idx + 1} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        if metrics['f1'] > best_mm_f1:
            best_mm_f1 = metrics['f1']
            best_mm_model = pipeline

    multimodal_results = aggregate_cv_results(multimodal_fold_metrics)
    save_results(multimodal_results, 'multimodal_model')
    if best_mm_model:
        save_multimodal_model(best_mm_model)

    # ============ COMPARISON ============
    print_model_comparison({
        'Text (DistilBERT)': text_results,
        'Audio (SVM)': audio_results,
        'Multimodal (Fusion)': multimodal_results
    })

    # Save combined comparison
    comparison = {
        'text_model': text_results,
        'audio_model': audio_results,
        'multimodal_model': multimodal_results
    }
    save_results(comparison, 'model_comparison')

    print("\n✓ All models trained and saved successfully!")
    print(f"  Models saved in: {MODELS_DIR}")
    print(f"  Results saved in: {RESULTS_DIR}")


if __name__ == '__main__':
    train_and_evaluate_all()
