import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer



def validate_config(config: Dict[str, Any]) -> None:
    """Блокує потрапляння protected_attributes до features."""
    protected = set(config.get("protected_attributes", []))
    features = set()
    for group in config.get("features", {}).values():
        if group:
            features.update(group)
    leaked = protected & features
    if leaked:
        raise ValueError(f"Захищені ознаки потрапили у features: {leaked}")


class Winsorizer(BaseEstimator, TransformerMixin):
    """Обчислює квантилі виключно на fit (X_train) без витоку даних на тесті."""
    def __init__(self, lower_quantile: float = 0.05, upper_quantile: float = 0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.lower_bounds_ = np.percentile(X_arr, self.lower_quantile * 100, axis=0)
        self.upper_bounds_ = np.percentile(X_arr, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


class AdaptiveImputer(BaseEstimator, TransformerMixin):
    """Маршрутизує колонки за часткою пропусків згідно з політикою з конфігурації."""
    def __init__(self, policy: List[Dict]):
        self.policy = policy
        self.audit_trail_ = {}
        self.imputers_ = []

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.imputers_ = []
        self.audit_trail_ = {}
        n_cols = X_arr.shape[1]

        for col_idx in range(n_cols):
            col_data = X_arr[:, col_idx]
            # Визначаємо частку пропусків
            if col_data.dtype.kind in ('O', 'U', 'S'):
                missing_mask = pd.isna(col_data)
            else:
                missing_mask = np.isnan(col_data.astype(float))
            missing_share = float(np.mean(missing_mask))

            applied_rule = None
            for rule in sorted(self.policy, key=lambda r: r['max_share']):
                if missing_share <= rule['max_share']:
                    applied_rule = rule
                    break

            self.audit_trail_[col_idx] = {
                'missing_share': missing_share,
                'rule': applied_rule
            }

            imp = SimpleImputer(
                strategy=applied_rule['strategy'],
                fill_value=applied_rule.get('fill_value', None),
                add_indicator=applied_rule.get('add_indicator', False)
            )
            imp.fit(col_data.reshape(-1, 1))
            self.imputers_.append(imp)

        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        transformed_parts = []
        for col_idx, imp in enumerate(self.imputers_):
            col_data = X_arr[:, col_idx].reshape(-1, 1)
            transformed_parts.append(imp.transform(col_data))
        return np.hstack(transformed_parts)

# class ValueAliasMapper(BaseEstimator, TransformerMixin):
#     """Зводить синоніми категорій до єдиного стандарту (EDA знахідка)."""
#     def __init__(self):
#         self.mapping = {
#             'Phone': 'Mobile Phone',
#             'Mobile': 'Mobile Phone',
#             'CC': 'Credit Card',
#             'COD': 'Cash on Delivery'
#         }
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         X_df = pd.DataFrame(X)
#         return X_df.replace(self.mapping).values

class ValueAliasMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = {
            'Phone': 'Mobile Phone',
            'Mobile': 'Mobile Phone',
            'CC': 'Credit Card',
            'COD': 'Cash on Delivery'
        }
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.replace(self.mapping).values

def get_transformer(name: str):
    if name == 'log1p':
        return FunctionTransformer(np.log1p)
    if name == 'standard_scaler':
        return StandardScaler()
    if name == 'one_hot':
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    if name == 'ordinal_encoder':
        return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if name == 'winsorize':
        return Winsorizer(lower_quantile=0.05, upper_quantile=0.95)
    if name == 'aliases':
        return ValueAliasMapper()
    if name == 'passthrough':
        return FunctionTransformer(lambda x: x)
    return 'passthrough'

# def get_transformer(name: str):
#     if name == 'log1p':
#         return FunctionTransformer(np.log1p)
#     if name == 'standard_scaler':
#         return StandardScaler()
#     if name == 'one_hot':
#         return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     if name == 'ordinal_encoder':
#         return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#     if name == 'winsorize':
#         return Winsorizer(lower_quantile=0.05, upper_quantile=0.95)
#     return 'passthrough'


class DynamicPipelineBuilder:
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        validate_config(self.config)

    def build(self) -> ColumnTransformer:
        """Повертає ненавчений ColumnTransformer. До даних не звертається."""
        model_family = self.config["model_family"]
        transform_policy = self.config["transform_policy"][model_family]
        imputation_policy = self.config["imputation_policy"]

        transformers = []
        for group_name, cols in self.config.get("features", {}).items():
            if not cols:
                continue

            imp_policy = imputation_policy['numeric'] if 'numeric' in group_name else imputation_policy['categorical']
            steps = [('imputer', AdaptiveImputer(imp_policy))]

            for t_name in transform_policy.get(group_name, []):
                steps.append((t_name, get_transformer(t_name)))

            transformers.append((group_name, Pipeline(steps), cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")