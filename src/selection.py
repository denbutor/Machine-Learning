import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
import warnings

warnings.filterwarnings('ignore')


class MultilevelFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Крок 5: Багаторівнева воронка відбору ознак.
    1. Filter: грубе скорочення за допомогою mutual_info (до k=30).
    2. Hybrid/Wrapper: mlxtend SFS (скорочення до k=15).
    3. Embedded: відкидання ознак з нульовими вагами через L1-регуляризацію (Lasso).
    """

    def __init__(self, filter_k=30, sfs_k=15, cv=3, random_state=42):
        self.filter_k = filter_k
        self.sfs_k = sfs_k
        self.cv = cv
        self.random_state = random_state

        self.filter_selector = SelectKBest(score_func=mutual_info_classif, k=self.filter_k)
        self.sfs = None
        self.l1_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500,
                                           random_state=self.random_state)
        self.final_features_idx_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        # Якщо вхід DataFrame, запам'ятовуємо імена колонок
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
        else:
            self.feature_names_in_ = np.array([f"f_{i}" for i in range(X.shape[1])])

        X_arr = np.array(X)
        y_arr = np.array(y)

        # 1. Filter Methods (Грубе скорочення)
        # Якщо вхідних ознак менше за filter_k, беремо всі
        actual_filter_k = min(self.filter_k, X_arr.shape[1])
        self.filter_selector.set_params(k=actual_filter_k)
        X_filtered = self.filter_selector.fit_transform(X_arr, y_arr)
        filter_idx = self.filter_selector.get_support(indices=True)

        # 2. Wrapper (Hybrid SFS через mlxtend)
        # SFS вимагає базової моделі, беремо швидку логістичну регресію
        base_lr = LogisticRegression(max_iter=500, random_state=self.random_state)
        actual_sfs_k = min(self.sfs_k, X_filtered.shape[1])

        self.sfs = SequentialFeatureSelector(
            base_lr,
            k_features=actual_sfs_k,
            forward=True,
            floating=False,
            scoring='roc_auc',
            cv=self.cv,
            n_jobs=-1
        )
        self.sfs.fit(X_filtered, y_arr)
        # Індекси ознак після SFS (відносно X_filtered)
        sfs_idx_relative = list(self.sfs.k_feature_idx_)
        # Індекси ознак після SFS (відносно оригінального X)
        sfs_idx_absolute = filter_idx[sfs_idx_relative]

        # 3. Embedded Methods (L1 Регуляризація)
        X_sfs = X_arr[:, sfs_idx_absolute]
        self.l1_model.fit(X_sfs, y_arr)

        # Відбираємо ті, де коефіцієнти не обнулилися
        l1_support = np.abs(self.l1_model.coef_[0]) > 1e-5
        self.final_features_idx_ = sfs_idx_absolute[l1_support]

        # 4. Tree-based Importance (фінальне ранжування за Gini)
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        if len(self.final_features_idx_) > 0:
            X_final = X_arr[:, self.final_features_idx_]
            rf.fit(X_final, y_arr)
            self.feature_importances_ = rf.feature_importances_
        else:
            self.feature_importances_ = np.array([])

        return self

    def transform(self, X):
        X_arr = np.array(X)
        if self.final_features_idx_ is None or len(self.final_features_idx_) == 0:
            return X_arr  # Fallback, якщо все обнулилося
        return X_arr[:, self.final_features_idx_]

    def get_feature_names_out(self, input_features=None):
        if self.final_features_idx_ is None:
            return self.feature_names_in_
        return self.feature_names_in_[self.final_features_idx_]