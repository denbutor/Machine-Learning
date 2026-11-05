import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold


class TenureStage(BaseEstimator, TransformerMixin):
    """
    Категорія А: Часові ознаки.
    Бінінг стажу клієнта на основі експертних меж відтоку.
    """

    def __init__(self, col_name='Tenure'):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        bins = [-1, 1, 6, 12, 100]
        labels = ['New', 'Early', 'Established', 'Loyal']

        # Якщо вхідний масив є numpy, перетворимо на DataFrame і візьмемо потрібний стовпець
        # У пайплайні ми передамо його вже після відбору колонок
        if isinstance(X_df.iloc[:, 0].values[0], (int, float)):
            col = X_df.iloc[:, 0]
        else:
            col = X_df[self.col_name]

        return pd.cut(col, bins=bins, labels=labels, right=True).astype(str).to_frame().values


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Категорія А: Циклічне кодування (sin/cos).
    """

    def __init__(self, max_val):
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=float)
        sin_x = np.sin(2 * np.pi * X_arr / self.max_val)
        cos_x = np.cos(2 * np.pi * X_arr / self.max_val)
        return np.hstack([sin_x, cos_x])


class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Категорія Б: Взаємодії.
    Обчислює бізнес-метрики з урахуванням ділення на нуль (ε = 1e-5).
    """

    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=['CashbackAmount', 'OrderCount', 'CouponUsed', 'Tenure', 'DaySinceLastOrder'])

        cashback_per_order = X_df['CashbackAmount'] / (X_df['OrderCount'] + self.epsilon)
        coupon_per_order = X_df['CouponUsed'] / (X_df['OrderCount'] + self.epsilon)
        orders_per_month = X_df['OrderCount'] / (X_df['Tenure'] + 1)
        recency_ratio = X_df['DaySinceLastOrder'] / (X_df['Tenure'] + 1)

        return np.column_stack([cashback_per_order, coupon_per_order, orders_per_month, recency_ratio])


class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """
    Категорія Б: Трансформація.
    Стабілізує дисперсію та дозволяє нулі (альтернатива log1p).
    """

    def __init__(self):
        self.pt = PowerTransformer(method='yeo-johnson', standardize=False)

    def fit(self, X, y=None):
        self.pt.fit(X)
        return self

    def transform(self, X):
        return self.pt.transform(X)


class OOFWoEEncoder(BaseEstimator, TransformerMixin):
    """
    Категорія В: Кодування високої кардинальності.
    Регуляризований Out-Of-Fold WoE для захисту від витоку (перенавчання на рідкісних сегментах).
    """

    def __init__(self, cv=5, smoothing=0.5):
        self.cv = cv
        self.smoothing = smoothing
        self.global_woe_ = {}
        self.overall_prob_ = None

    def fit(self, X, y):
        # Навчаємо глобальний словник для використання під час transform (на test_set)
        y_sr = pd.Series(y)
        self.overall_prob_ = y_sr.mean()

        X_sr = pd.Series(X.ravel())
        counts = X_sr.value_counts()
        pos_counts = y_sr.groupby(X_sr).sum()

        p_event = (pos_counts + self.smoothing) / (y_sr.sum() + 2 * self.smoothing)
        p_non_event = ((counts - pos_counts) + self.smoothing) / ((len(y_sr) - y_sr.sum()) + 2 * self.smoothing)

        self.global_woe_ = np.log(p_event / p_non_event).to_dict()
        return self

    def fit_transform(self, X, y):
        # Обчислює OOF WoE для тренувальних даних
        self.fit(X, y)
        y_sr = pd.Series(y)
        X_sr = pd.Series(X.ravel())
        oof_encoded = np.zeros(len(X))

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X_sr.iloc[train_idx], y_sr.iloc[train_idx]

            counts = X_train.value_counts()
            pos_counts = y_train.groupby(X_train).sum()

            p_e = (pos_counts + self.smoothing) / (y_train.sum() + 2 * self.smoothing)
            p_n = ((counts - pos_counts) + self.smoothing) / ((len(y_train) - y_train.sum()) + 2 * self.smoothing)

            fold_woe = np.log(p_e / p_n).to_dict()
            oof_encoded[val_idx] = X_sr.iloc[val_idx].map(fold_woe).fillna(0)

        return oof_encoded.reshape(-1, 1)

    def transform(self, X):
        # На тестових даних використовуємо глобальні ваги
        X_sr = pd.Series(np.array(X).ravel())
        return X_sr.map(self.global_woe_).fillna(0).values.reshape(-1, 1)