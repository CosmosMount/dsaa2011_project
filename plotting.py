import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from training import perform_clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform


def plot_tsne(
    X_pca,
    y,
    perplexities=(30,),
    random_state=42,
    figsize=(6, 5),
    compute_metrics=True,
):
    """
    X_pca: PCA-transformed features (n_samples, n_components)
    y: labels
    perplexities: list of perplexity values
    """

    results = {}

    n_plots = len(perplexities)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize[0]*n_plots, figsize[1]))

    if n_plots == 1:
        axes = [axes]

    for ax, p in zip(axes, perplexities):

        # --- t-SNE ---
        tsne = TSNE(
            n_components=2,
            perplexity=p,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        emb = tsne.fit_transform(X_pca)

        df = pd.DataFrame(emb, columns=["tsne1", "tsne2"])
        df["label"] = y

        # --- plot ---
        sns.scatterplot(
            data=df,
            x="tsne1",
            y="tsne2",
            hue="label",
            palette="Set2",
            s=25,
            alpha=0.8,
            ax=ax,
            legend=(p == perplexities[0])
        )

        ax.set_title(f"t-SNE (perplexity={p})")

        metrics = {}

        if compute_metrics:
            # centroid
            centroids = df.groupby("label")[["tsne1", "tsne2"]].mean()

            # spread
            def compute_spread(group):
                center = group[["tsne1", "tsne2"]].mean()
                return ((group[["tsne1", "tsne2"]] - center) ** 2).sum(axis=1).mean()

            spread = df.groupby("label").apply(compute_spread)

            # distance
            dist_matrix = pd.DataFrame(
                squareform(pdist(centroids)),
                index=centroids.index,
                columns=centroids.index
            )

            print(f"\n=== Perplexity = {p} ===")
            print("Centroid distance matrix:")
            print(dist_matrix.round(3))
            print("Spread:")
            print(spread.round(3))

            metrics = {
                "centroids": centroids,
                "spread": spread,
                "distance_matrix": dist_matrix
            }

        results[p] = {
            "embedding": df,
            "metrics": metrics
        }

    plt.tight_layout()
    plt.show()

    return results

def plot_roc_auc(models, X_test, y_test, title="ROC AUC"):
    """
    Plots ROC curves for given models on test data.
    models: dict of {name: fitted_model}
    X_test, y_test: test data
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    classes = np.unique(y_test)
    is_multiclass = len(classes) > 2

    plt.figure(figsize=(7, 5))
    any_curve = False

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            continue

        if not is_multiclass:
            y_score = scores[:, 1] if scores.ndim > 1 else scores
            fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=classes[1])
            roc_auc = auc(fpr, tpr)
        else:
            if scores.ndim == 1:
                continue
            y_bin = label_binarize(y_test, classes=classes)
            fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
            roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        any_curve = True

    if any_curve:
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    else:
        print("No models with probability or decision scores available for ROC AUC plot.")

def plot_cluster_comparison(X_cluster, X_plot, methods, n_clusters, title_prefix):

    if isinstance(X_plot, pd.DataFrame):
        x_vals = X_plot.iloc[:, 0].values
        y_vals = X_plot.iloc[:, 1].values
    else:
        x_vals = X_plot[:, 0]
        y_vals = X_plot[:, 1]

    n = len(methods)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (method, kwargs) in zip(axes, methods.items()):
        _, labels = perform_clustering(
            X_cluster, method=method, n_clusters=n_clusters, **kwargs
        )
        ax.scatter(x_vals, y_vals, c=labels, cmap="tab10", s=18, alpha=0.8)
        ax.set_title(f"{title_prefix} - {method}")
        ax.set_xlabel("dim1")
        ax.set_ylabel("dim2")

    for ax in axes[len(methods):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()