import pandas as pd
import numpy as np

def _safe_class_key(label):
    label = str(label)
    return (
        label.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


def evaluate_classification(model, X_test, y_test):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    
    y_pred = model.predict(X_test)
    classes = np.unique(y_test)
    is_multiclass = len(classes) > 2

    avg = 'macro'
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=avg, zero_division=0),
        'recall': recall_score(y_test, y_pred, average=avg),
        'f1': f1_score(y_test, y_pred, average=avg),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # Per-class F1 for multi-class reporting
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    for label, score in zip(classes, per_class_f1):
        results[f"f1_{_safe_class_key(label)}"] = score

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if is_multiclass:
            results['roc_auc'] = roc_auc_score(
                y_test, y_proba, multi_class='ovr', average='macro'
            )
        else:
            results['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])

    return results

def evaluate_clustering(X, labels, y_true=None):

    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        adjusted_rand_score,
        normalized_mutual_info_score
    )

    results = {}

    # ---------- Internal metrics ----------
    if len(set(labels)) > 1:

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        db_inv = 1 / (1 + db)

        results.update({
            'silhouette': sil,
            'calinski_harabasz': ch,
            'davies_bouldin': db,
            'db_inv': db_inv
        })

        sil_n = (sil + 1) / 2                # [-1,1] → [0,1]
        ch_n = np.log1p(ch) / 10             # log 
        db_n = db_inv                        # [0,1]

        internal_score = np.mean([sil_n, ch_n, db_n])

        results['internal_score'] = internal_score

    # ---------- External metrics ----------
    if y_true is not None:

        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        ari_n = (ari + 1) / 2     # [-1,1] → [0,1]
        nmi_n = nmi               # [0,1]

        external_score = np.mean([ari_n, nmi_n])

        results.update({
            'ARI': ari,
            'NMI': nmi,
            'external_score': external_score
        })

    # ---------- Final combined score ----------
    if 'internal_score' in results and 'external_score' in results:
        results['score'] = 0.4 * results['internal_score'] + 0.6 * results['external_score']

    elif 'internal_score' in results:
        results['score'] = results['internal_score']

    return results