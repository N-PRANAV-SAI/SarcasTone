import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sarcastone.config import *


def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics for binary classification."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'report': classification_report(y_true, y_pred, output_dict=True, digits=4,
                                         target_names=['Not Sarcastic', 'Sarcastic'])
    }


def aggregate_cv_results(fold_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across cross-validation folds."""
    metric_keys = ['accuracy', 'f1']

    aggregated = {}
    for key in metric_keys:
        values = [fold[key] for fold in fold_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'per_fold': values
        }

    # Also aggregate precision and recall from reports
    for avg_type in ['weighted avg']:
        for metric in ['precision', 'recall', 'f1-score']:
            safe_key = f"{avg_type}_{metric}".replace(' ', '_').replace('-', '_')
            values = []
            for fold in fold_metrics:
                if 'report' in fold and avg_type in fold['report']:
                    values.append(fold['report'][avg_type][metric])
            if values:
                aggregated[safe_key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'per_fold': values
                }

    # Aggregate confusion matrices
    cms = [np.array(fold['confusion_matrix']) for fold in fold_metrics]
    aggregated['confusion_matrix_sum'] = np.sum(cms, axis=0).tolist()
    aggregated['confusion_matrix_mean'] = np.mean(cms, axis=0).tolist()

    aggregated['n_folds'] = len(fold_metrics)
    aggregated['fold_metrics'] = fold_metrics

    return aggregated


def save_results(results: Dict, model_name: str, results_dir: str = None):
    """Save evaluation results to JSON."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    path = os.path.join(results_dir, f'{model_name}_results.json')

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"Results saved to {path}")
    return path


def load_results(model_name: str, results_dir: str = None) -> Optional[Dict]:
    """Load evaluation results from JSON."""
    if results_dir is None:
        results_dir = RESULTS_DIR

    path = os.path.join(results_dir, f'{model_name}_results.json')

    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)


def print_model_comparison(results_dict: Dict[str, Dict]):
    """Print a comparison table of all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - 5-Fold Stratified Cross-Validation")
    print("=" * 80)

    header = f"{'Model':<25} {'Accuracy':>15} {'F1-Score':>15}"
    print(header)
    print("-" * 80)

    for model_name, results in results_dict.items():
        if 'accuracy' in results and isinstance(results['accuracy'], dict):
            acc = f"{results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}"
            f1 = f"{results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}"
        else:
            acc = f"{results.get('accuracy', 0):.4f}"
            f1 = f"{results.get('f1', 0):.4f}"

        print(f"{model_name:<25} {acc:>15} {f1:>15}")

    print("=" * 80)
