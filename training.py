def train_classification(
        X_train, 
        y_train, 
        method='rf', 
        preprocessor=None,
        use_smote=False, 
        **kwargs
    ):
    """
    Classification Methods
    """
    from sklearn.linear_model import LogisticRegression, Perceptron
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        ExtraTreesClassifier, 
        HistGradientBoostingClassifier
    )
    from sklearn.pipeline import Pipeline
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    method = method.lower()
    
    if method == 'logistic':
        params = {
            "class_weight": "balanced"
        }
        model = LogisticRegression(**params, **kwargs)

    elif method == 'perceptron':
        model = Perceptron(**kwargs)

    elif method == 'svm':
        params = {
            "probability": True,
            "class_weight": "balanced"
        }
        model = SVC(**params, **kwargs)

    elif method == 'decision_tree':
        model = DecisionTreeClassifier(**kwargs)

    elif method == 'random_forest':
        params = {
            "class_weight": "balanced",
            "random_state": 42
        }
        model = RandomForestClassifier(**params, **kwargs)

    elif method == 'gradient_boosting':
        model = GradientBoostingClassifier(**kwargs)

    elif method == 'adaboost':
        params = {
            "random_state": 42
        }
        model = AdaBoostClassifier(**params, **kwargs)

    elif method == 'xgboost':
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
        }
        final_params = params.copy()
        final_params.update(kwargs)
        model = XGBClassifier(**final_params)

    elif method == 'lightgbm':
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42
        }
        final_params = params.copy()
        final_params.update(kwargs)
        model = LGBMClassifier(**final_params)

    elif method == 'catboost':
        params = {
            "loss_function": "MultiClass",
            "verbose": 0,
            "random_state": 42
        }
        final_params = params.copy()
        final_params.update(kwargs)
        model = CatBoostClassifier(**final_params)
    
    elif method == 'extra_trees':
        params = {
            "n_estimators": 200,
            "max_features": "sqrt",
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        final_params = params.copy()
        final_params.update(kwargs)
        model = ExtraTreesClassifier(**final_params)
    
    elif method == 'hist_gradient_boosting':
        params = {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_depth": None,
            "l2_regularization": 1.0,
            "random_state": 42
        }
        final_params = params.copy()
        final_params.update(kwargs)

        model = HistGradientBoostingClassifier(**final_params)

    else:
        raise ValueError(f"Unexpected Method: {method}")

    steps = []

    if preprocessor is not None:
        steps.append(('preprocessor', preprocessor))

    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
        pipeline = ImbPipeline(steps + [('model', model)])
    else:
        pipeline = Pipeline(steps + [('model', model)])

    pipeline.fit(X_train, y_train)
    return pipeline
    
def perform_clustering(X, method='kmeans', n_clusters=3, **kwargs):
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
    from sklearn.metrics.pairwise import rbf_kernel
    
    method = method.lower()

    # ---------- k-Means ----------
    if method == 'kmeans':
        model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            random_state=42,
            **kwargs
        )
        labels = model.fit_predict(X)

    # ---------- Kernel k-Means（近似实现） ----------
    elif method == 'kernel_kmeans':
        gamma = kwargs.get('gamma', 1.0)

        K = rbf_kernel(X, gamma=gamma)  # 核矩阵
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(K)

    # ---------- DBSCAN ----------
    elif method == 'dbscan':
        model = DBSCAN(**kwargs)
        labels = model.fit_predict(X)

    # ---------- OPTICS ----------
    elif method == 'optics':
        model = OPTICS(**kwargs)
        labels = model.fit_predict(X)

    # ---------- Agglomerative ----------
    elif method == 'agglomerative':
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            **kwargs
        )
        labels = model.fit_predict(X)

    # ---------- GMM ----------
    elif method == 'gmm':
        model = GaussianMixture(
            n_components=n_clusters,
            random_state=42,
            **kwargs
        )
        model.fit(X)
        labels = model.predict(X)

    else:
        raise ValueError(f"Unexpected Method: {method}")

    return model, labels

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

from sklearn.model_selection import cross_val_score, StratifiedKFold

def _train_eval(
    params,
    X_train, y_train,
    X_val, y_val,
    preprocessor,
    method,
    use_cv=False,
    use_smote=False,
    cv=5,
    scoring='f1_macro',
):

    model = train_classification(
        X_train,
        y_train,
        preprocessor=preprocessor,
        method=method,
        use_smote=use_smote,
        **params
    )

    if use_cv:
        cv_strategy = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=42
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=1
        )

        return scores.mean(), model, params

    else:
        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average='macro')
        return score, model, params


def grid_search_classification(
    X_train, y_train,
    X_val=None, y_val=None,
    preprocessor=None,
    method='rf',
    param_grid=None,
    n_jobs=-1,
    use_cv=False,
    use_smote=False,
    cv=5
):

    results = Parallel(n_jobs=n_jobs)(
        delayed(_train_eval)(
            params,
            X_train, y_train,
            X_val, y_val,
            preprocessor,
            method,
            use_cv,
            use_smote,
            cv
        )
        for params in ParameterGrid(param_grid)
    )

    best_score = -1
    best_model = None
    best_params = None

    for score, model, params in results:
        metric_name = "CV_F1" if use_cv else "VAL_F1"
        print(f"{method} | {params} | {metric_name}={score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    if use_cv:
        best_model.fit(X_train, y_train)

    return best_model, best_params, best_score

def train_ensemble_voting(
    X_train, y_train,
    methods,
    preprocessor=None,
    voting='soft'
):
    from sklearn.ensemble import VotingClassifier
    models = []

    for m in methods:
        model = train_classification(
            X_train,
            y_train,
            method=m,
            preprocessor=preprocessor
        )
        models.append((m, model))

    ensemble = VotingClassifier(
        estimators=models,
        voting=voting,
        n_jobs=-1
    )

    ensemble.fit(X_train, y_train)
    return ensemble