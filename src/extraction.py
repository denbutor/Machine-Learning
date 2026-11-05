import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.io import imread
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TextExtractor(BaseEstimator, TransformerMixin):
    """
    Крок 2: Екстрактор текстових ознак (Amazon Reviews).
    TF-IDF (500 слів) -> TruncatedSVD (5 тем).
    Стійкий до порожніх входів (NaN).
    """

    def __init__(self, max_features=500, n_components=5):
        self.max_features = max_features
        self.n_components = n_components
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features, stop_words='english')),
            ('svd', TruncatedSVD(n_components=self.n_components, random_state=42))
        ])

    def fit(self, X, y=None):
        # Замінюємо NaN на порожні рядки перед навчанням
        X_filled = pd.Series(X).fillna("")
        self.pipeline.fit(X_filled)
        return self

    def transform(self, X):
        X_filled = pd.Series(X).fillna("")
        # Якщо вхід порожній, TF-IDF -> SVD видасть нулі або близькі до нуля значення,
        # що є коректною поведінкою і не "ламає" конвеєр.
        return self.pipeline.transform(X_filled)


class ImageExtractor(BaseEstimator, TransformerMixin):
    """
    Крок 2: Екстрактор ознак із зображень (GLCM текстурні ознаки).
    Рахує contrast, correlation, energy, homogeneity.
    Стійкий до відсутніх файлів (повертає NaN).
    """

    def __init__(self, image_dir='data/images'):
        self.image_dir = image_dir

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for filename in X:
            # Якщо ім'я файлу порожнє або це не рядок
            if pd.isna(filename) or not isinstance(filename, str):
                features.append([np.nan] * 4)
                continue

            filepath = os.path.join(self.image_dir, filename)
            if not os.path.exists(filepath):
                features.append([np.nan] * 4)
                continue

            try:
                # Читаємо зображення
                img = imread(filepath)

                # Якщо кольорове (RGB) – переводимо в градації сірого
                if len(img.shape) == 3:
                    img = rgb2gray(img)
                elif len(img.shape) == 4:  # RGBA
                    img = rgb2gray(img[..., :3])

                # Конвертуємо у 8-бітний формат для graycomatrix
                img_uint8 = (img * 255).astype(np.uint8)

                # Рахуємо матрицю GLCM
                glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

                # Дістаємо ознаки
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

                features.append([contrast, correlation, energy, homogeneity])

            except Exception:
                # Якщо фото бите, повертаємо пропуски
                features.append([np.nan] * 4)

        return np.array(features)