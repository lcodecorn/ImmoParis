"""
run.py
======
Point d'entrée unique du projet.

    python run.py

Enchaîne dans l'ordre :
  1. Téléchargement & nettoyage des données  (prepare_data)
  2. Feature engineering partagé             (build_features)
  3. Entraînement CatBoost                   (train_catboost)
  4. Entraînement Random Forest sklearn      (train_random_forest)
  5. Entraînement Random Forest NumPy        (train_numpy_rf)

Sorties :
  Data/xp.csv
  Data/metro.csv
  Model/model_cat.pkl              +  Model/preprocessor_cat.pkl
  Model/model_random_forest.pkl    +  Model/preprocessor_random_forest.pkl
  Model/model_numpy.pkl            +  Model/preprocessor_numpy.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import zipfile
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = "Data"
MODEL_DIR  = "Model"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# Téléchargement & nettoyage des données
DVF_BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv"
DVF_YEARS          = [2020, 2021, 2022, 2023, 2024, 2025]
PARIS_COMMUNES = [f"751{str(i).zfill(2)}" for i in range(1, 21)]
METRO_SOURCE_URL = ("https://static.data.gouv.fr/resources/lignes-et-stations-de-metro-en-france/20250218-105659/metro-france.csv")
METRO_SOURCE_FILE  = os.path.join(DATA_DIR, "metro-france.csv")



def download_dvf():
    for year in DVF_YEARS:
        year_dfs = []

        for commune in PARIS_COMMUNES:
            filename = f"{commune}_{year}.csv"
            dest = os.path.join(DATA_DIR, filename)

            if os.path.exists(dest):
                print(f"  {filename} déjà présent, skip.")
            else:
                url = f"{DVF_BASE_URL}/{year}/communes/75/{commune}.csv"
                print(f"  Téléchargement {filename} ...")

                try:
                    r = requests.get(url)
                    r.raise_for_status()

                    with open(dest, "wb") as f:
                        f.write(r.content)

                    print(f"  → {filename} téléchargé.")

                except requests.exceptions.HTTPError:
                    print(f"  ⚠️ Fichier manquant: {url}")
                    continue

            # Load immediately to concat per year
            try:
                df = pd.read_csv(dest, low_memory=False)
                year_dfs.append(df)
            except Exception as e:
                print(f"  ⚠️ Erreur lecture {filename}: {e}")

        # concat ALL arrondissements for that year
        if year_dfs:
            df_year = pd.concat(year_dfs, ignore_index=True)
            year_path = os.path.join(DATA_DIR, f"75_{year}.csv")
            df_year.to_csv(year_path, index=False)
            print(f"  ✅ Année {year} concaténée → {year_path}")


def preprocess_dvf():
    dfs = []
    for year in DVF_YEARS:
        path = os.path.join(DATA_DIR, f"75_{year}.csv")
        dfs.append(pd.read_csv(path, low_memory=False))

    df = pd.concat(dfs, axis=0, ignore_index=True)

    col_to_keep = [
        "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
        "adresse_nom_voie", "code_postal", "type_local", "surface_reelle_bati",
        "nombre_pieces_principales", "longitude", "latitude",
    ]
    df = df[col_to_keep]
    df = df[df["type_local"]       == "Appartement"]
    df = df[df["nature_mutation"]  == "Vente"]
    df = df[df["surface_reelle_bati"] > 9]
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].replace(0, 1)
    df["valeur_fonciere"] = df.groupby("code_postal")["valeur_fonciere"].transform(
        lambda x: x.fillna(x.median())
    )
    df = df.dropna()
    df["price_per_sqrtm"] = df["valeur_fonciere"] / df["surface_reelle_bati"]

    Q1 = df["price_per_sqrtm"].quantile(0.25)
    Q3 = df["price_per_sqrtm"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["price_per_sqrtm"] >= Q1 - 1.5 * IQR) & (df["price_per_sqrtm"] <= Q3 + 1.0 * IQR)]
    df = df.drop(columns=["type_local", "nature_mutation"])

    out = os.path.join(DATA_DIR, "xp.csv")
    df.to_csv(out, index=False)
    print(f"  → DVF exporté : {out}  ({df.shape[0]:,} lignes)")


def download_metro():
    if os.path.exists(METRO_SOURCE_FILE):
        print("  Fichier métro déjà présent, skip.")
        return

    print("  Téléchargement métro-france.csv ...")

    r = requests.get(METRO_SOURCE_URL)
    r.raise_for_status()

    with open(METRO_SOURCE_FILE, "wb") as f:
        f.write(r.content)

    print(f"  → {METRO_SOURCE_FILE} téléchargé.")

def preprocess_metro():
    df2 = pd.read_csv(METRO_SOURCE_FILE, sep=",", low_memory=False)
    print(df2.columns.tolist())
    df2 = df2[df2["Commune nom"].str.startswith("Paris", na=False)]
    df2 = df2.drop(columns=["ID Line", "Commune nom"])
    df2["Commune code Insee"] = (
        df2["Commune code Insee"].astype(str).str[:-3]
        + "0"
        + df2["Commune code Insee"].astype(str).str[-2:]
    ).astype(int)
    out = os.path.join(DATA_DIR, "metro.csv")
    df2.to_csv(out, index=False)
    print(f"  → Métro exporté : {out}  ({df2.shape[0]} stations)")


def prepare_data():
    print("\n" + "=" * 60)
    print("1 — Préparation des données")
    print("=" * 60)
    download_dvf()
    preprocess_dvf()
    download_metro()
    preprocess_metro()


# Feature engineering
CENTER_LON, CENTER_LAT = 2.3384444444444446, 48.86152777777778
N_CLUSTERS = 20
AGG_SPEC = {
    'price_per_sqrtm': 'median',
    'valeur_fonciere': 'median',
    'nombre_pieces_principales': 'median',
    'surface_reelle_bati': 'median',
    'surface_per_piece': 'median',
    'longitude': 'mean',
    'latitude': 'mean',
    'days_since_start': 'median',
    'geo_cluster': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
    'dist_center': 'median',
    'nearest_metro_dist_km': 'median',
    'station_tx_count': 'median',
    'station_avg_surface': 'median',
    'station_surface_std': 'median',
    'station_median_surface': 'median',
    'station_avg_rooms': 'median',
    'station_rooms_std': 'median',
    'station_median_rooms': 'median',
    'station_surface_range': 'median',
    'surface_vs_station_avg': 'median',
    'rooms_vs_station_avg': 'median',
    'larger_than_station_median': 'median',
    'metro_count_300m': 'median',
    'metro_count_500m': 'median',
    'very_close_to_metro': 'median',
    'is_studio': 'median',
    'is_large': 'median',
    'date_mutation': 'first',
    'transaction_count': 'median',
    'total_transactions': 'median',
    'market_activity_ratio': 'median'
}


def build_features():
    """
    Charge xp.csv + metro.csv, applique tout le feature engineering commun
    aux deux modèles, et retourne (train_agg, test_agg, km, station_stats_train).
    """
    print("\n" + "=" * 60)
    print("2 — Feature engineering")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, "xp.csv"))
    metro_df = pd.read_csv(os.path.join(DATA_DIR, "metro.csv"))
    metro_df.columns = metro_df.columns.str.strip()
    print(f"  DVF : {df.shape}  |  Métro : {metro_df.shape}")

    # Temps
    print("\n  Features temporelles ...")
    df['date_mutation']   = pd.to_datetime(df['date_mutation'])
    df['year']            = df['date_mutation'].dt.year
    df['month']           = df['date_mutation'].dt.month
    df['day_of_week']     = df['date_mutation'].dt.dayofweek
    df['days_since_start'] = (df['date_mutation'] - df['date_mutation'].min()).dt.days

    # Monthly transaction count
    tx_counts = df.groupby(['code_postal', 'year', 'month']).size().reset_index(name='transaction_count')
    df = df.merge(tx_counts, on=['code_postal', 'year', 'month'], how='left')

    # Total historical transactions per postal code
    total_tx = df.groupby('code_postal').size().reset_index(name='total_transactions')
    df = df.merge(total_tx, on='code_postal', how='left')

    # Market activity ratio
    df['market_activity_ratio'] = df['transaction_count'] / df['total_transactions']
    
    # Géo
    print("  Features géographiques ...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['geo_cluster'] = km.fit_predict(df[['longitude', 'latitude']])
    df['dist_center'] = (
        np.sqrt((df['longitude'] - CENTER_LON) ** 2 +
                (df['latitude']  - CENTER_LAT)  ** 2) * 111
    )

    # Métro
    print("  Features métro (BallTree) ...")
    metro_coords  = np.radians(metro_df[['Latitude', 'Longitude']].values)
    tree          = BallTree(metro_coords, metric='haversine')
    appart_coords = np.radians(df[['latitude', 'longitude']].values)

    distances, indices = tree.query(appart_coords, k=1)
    df['nearest_metro_dist_km']  = distances.flatten() * 6371
    df['nearest_metro_station']  = metro_df.iloc[indices.flatten()]['Libelle station'].values
    df['nearest_metro_line']     = metro_df.iloc[indices.flatten()]['Libelle Line'].values
    df['metro_count_300m']       = [len(i) for i in tree.query_radius(appart_coords, r=0.3 / 6371)]
    df['metro_count_500m']       = [len(i) for i in tree.query_radius(appart_coords, r=0.5 / 6371)]
    df['very_close_to_metro']    = (df['nearest_metro_dist_km'] < 0.1).astype(int)

    # Appartement
    print("  Features appartement ...")
    df['surface_per_piece'] = df['surface_reelle_bati'] / df['nombre_pieces_principales'].replace(0, 1)
    df['is_studio']         = (df['nombre_pieces_principales'] == 1).astype(int)
    df['is_large']          = (df['nombre_pieces_principales'] >= 4).astype(int)
    df['surface_category']  = pd.cut(
        df['surface_reelle_bati'],
        bins=[9, 40, 80, float('inf')],
        labels=['small', 'medium', 'large'],
    )

    # Split temporel 80/20
    print("  Split temporel 80/20 ...")
    df_sorted    = df.sort_values('date_mutation').reset_index(drop=True)
    split_index  = int(len(df_sorted) * 0.8)
    train_df     = df_sorted.iloc[:split_index].copy()
    test_df      = df_sorted.iloc[split_index:].copy()
    print(f"  Train : {len(train_df):,}  |  Test : {len(test_df):,}  "
          f"(coupure : {test_df['date_mutation'].min().date()})")

    # Station stats (train only)
    print("  Station stats ...")
    station_stats_train = train_df.groupby('nearest_metro_station').agg(
        station_avg_surface    = ('surface_reelle_bati',          'mean'),
        station_surface_std    = ('surface_reelle_bati',          'std'),
        station_median_surface = ('surface_reelle_bati',          'median'),
        station_tx_count       = ('surface_reelle_bati',          'count'),
        station_avg_rooms      = ('nombre_pieces_principales',    'mean'),
        station_rooms_std      = ('nombre_pieces_principales',    'std'),
        station_median_rooms   = ('nombre_pieces_principales',    'median'),
    ).round(2)

    station_stats_train['station_surface_range'] = (
        station_stats_train['station_avg_surface'] + 2 * station_stats_train['station_surface_std'] -
        (station_stats_train['station_avg_surface'] - 2 * station_stats_train['station_surface_std'])
    )
    station_stats_train['station_surface_std'] = station_stats_train['station_surface_std'].replace(0, 1)
    station_stats_train['station_rooms_std']   = station_stats_train['station_rooms_std'].replace(0, 1)

    train_df = train_df.merge(station_stats_train, left_on='nearest_metro_station', right_index=True, how='left')
    test_df  = test_df.merge( station_stats_train, left_on='nearest_metro_station', right_index=True, how='left')

    station_cols = [c for c in train_df.columns if c.startswith('station_')]
    for col in station_cols:
        mean = train_df[col].mean()
        train_df[col] = train_df[col].fillna(mean)
        test_df[col]  = test_df[col].fillna(mean)

    # Features station
    for dataset in [train_df, test_df]:
        dataset['surface_vs_station_avg']    = (dataset['surface_reelle_bati']       - dataset['station_avg_surface']) / dataset['station_surface_std']
        dataset['rooms_vs_station_avg']      = (dataset['nombre_pieces_principales'] - dataset['station_avg_rooms'])   / dataset['station_rooms_std']
        dataset['larger_than_station_median'] = (dataset['surface_reelle_bati'] > dataset['station_median_surface']).astype(int)

    # Agr
    print("  Agrégation par (code_postal, year, month) ...")
    train_agg = train_df.groupby(['code_postal', 'year', 'month']).agg(AGG_SPEC).reset_index()
    test_agg  = test_df.groupby( ['code_postal', 'year', 'month']).agg(AGG_SPEC).reset_index()
    print(f"  Train agrégé : {train_agg.shape}  |  Test agrégé : {test_agg.shape}")

    return train_df, test_df, train_agg, test_agg, km, station_stats_train


# CatBoost
def train_catboost(train_df, test_df, train_agg, test_agg, km, station_stats_train):
    print("\n" + "=" * 60)
    print("3 — CatBoost")
    print("=" * 60)

    TARGET    = 'price_per_sqrtm'
    DROP_COLS = ['price_per_sqrtm', 'valeur_fonciere', 'date_mutation']


    x_train = train_agg.drop(columns=DROP_COLS)
    y_train = train_agg[TARGET]
    x_test  = test_agg.drop(columns=DROP_COLS)
    y_test  = test_agg[TARGET]

    categorical_features  = ['geo_cluster', 'year', 'month']
    cat_feature_indices   = [x_train.columns.get_loc(c) for c in categorical_features if c in x_train.columns]

    # Baseline
    print("\n  Baseline ...")
    baseline = CatBoostRegressor(
        iterations=1000, learning_rate=0.1, depth=6,
        l2_leaf_reg=3, random_seed=42, verbose=100, early_stopping_rounds=50,
    )
    baseline.fit(x_train, y_train, cat_features=cat_feature_indices,
                 eval_set=(x_test, y_test), use_best_model=True)
    _print_metrics("CatBoost Baseline", y_train, baseline.predict(x_train),
                                        y_test,  baseline.predict(x_test))

    # RandomizedSearchCV
    print("\n  Hyperparameter tuning ...")
    x_train_cat = x_train.copy()
    x_test_cat  = x_test.copy()
    for feat in categorical_features:
        x_train_cat[feat] = x_train_cat[feat].astype(str).astype('category')
        x_test_cat[feat]  = x_test_cat[feat].astype(str).astype('category')

    random_search = RandomizedSearchCV(
        estimator=CatBoostRegressor(random_seed=42, verbose=0, early_stopping_rounds=50),
        param_distributions={'iterations': [1000], 'learning_rate': [0.01],
                             'depth': [6], 'l2_leaf_reg': [3]},
        n_iter=20, cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2, random_state=42,
    )
    random_search.fit(x_train_cat, y_train, cat_features=categorical_features)
    best_params = random_search.best_params_
    print(f"  Meilleurs params : {best_params}")

    # Optimised model
    optimized = CatBoostRegressor(**best_params, random_seed=42, verbose=100, early_stopping_rounds=50)
    optimized.fit(x_train, y_train, cat_features=cat_feature_indices,
                  eval_set=(x_test, y_test), use_best_model=True)

    y_test_pred_opt = optimized.predict(x_test)
    _print_metrics("CatBoost Optimisé", y_train, optimized.predict(x_train),
                                        y_test,  y_test_pred_opt)

    # Save
    with open(os.path.join(MODEL_DIR, "model_cat.pkl"), "wb") as f:
        pickle.dump(optimized, f)
    with open(os.path.join(MODEL_DIR, "preprocessor_cat.pkl"), "wb") as f:
        pickle.dump({
            'kmeans_model':         km,
            'center_coords':        (CENTER_LON, CENTER_LAT),
            'station_stats':        station_stats_train,
            'categorical_features': categorical_features,
            'cat_feature_indices':  cat_feature_indices,
            'feature_columns':      list(x_train.columns),
            'drop_cols':            DROP_COLS,
        }, f)
    print("\n CatBoost sauvegardé → Model/model_cat.pkl + Model/preprocessor_cat.pkl")


# Random Forest
def train_random_forest(train_df, test_df, train_agg, test_agg):
    print("\n" + "=" * 60)
    print("4 — Random Forest")
    print("=" * 60)

    TARGET    = 'price_per_sqrtm'
    DROP_COLS = ['price_per_sqrtm', 'valeur_fonciere', 'date_mutation']


    x_train = train_agg.drop(columns=DROP_COLS)
    y_train = train_agg[TARGET]
    x_test  = test_agg.drop(columns=DROP_COLS)
    y_test  = test_agg[TARGET]

    # Preprocessor
    numeric_features   = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoric_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('ohe',    OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
        ]), categoric_features),
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('sc',     StandardScaler()),
        ]), numeric_features),
    ])
    x_train_proc = preprocessor.fit_transform(x_train)
    x_test_proc  = preprocessor.transform(x_test)

    # Baseline
    print("\n  Baseline ...")
    baseline = RandomForestRegressor(
        n_estimators=200, max_depth=30, min_samples_leaf=1,
        min_samples_split=15, random_state=42, n_jobs=-1, criterion='absolute_error',
    )
    baseline.fit(x_train_proc, y_train)
    _print_metrics("RF Baseline", y_train, baseline.predict(x_train_proc),
                                  y_test,  baseline.predict(x_test_proc))

    # GridSearchCV
    print("\n  GridSearchCV ...")
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid={
            'n_estimators':      [200, 250, 300],
            'max_depth':         [25, 30, 35],
            'min_samples_leaf':  [1, 2, 3],
            'min_samples_split': [15, 20, 25],
        },
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1,
    )
    grid_search.fit(x_train_proc, y_train)
    print(f"  Meilleurs params : {grid_search.best_params_}")
    print(f"  Best CV MAE      : {-grid_search.best_score_:.2f} €/m²")

    # Optimised model
    optimized       = grid_search.best_estimator_
    y_train_pred    = baseline.predict(x_train_proc)
    y_test_pred     = baseline.predict(x_test_proc)
    y_test_pred_opt = optimized.predict(x_test_proc)
    _print_metrics("RF Optimisé", y_train, optimized.predict(x_train_proc),
                                  y_test,  y_test_pred_opt)

    

    # Save
    with open(os.path.join(MODEL_DIR, "model_random_forest.pkl"), "wb") as f:
        pickle.dump(optimized, f)
    with open(os.path.join(MODEL_DIR, "preprocessor_random_forest.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    print("\n  RF sauvegardé → Model/model_random_forest.pkl + Model/preprocessor_random_forest.pkl")


# Random Forest NumPy (from scratch)

class KMeansNumPy:
    """K-Means clustering — pure NumPy"""

    def __init__(self, n_clusters=8, max_iter=300, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X, rng):
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_clusters(self, X, centroids):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, c in enumerate(centroids):
            distances[:, i] = np.sum((X - c) ** 2, axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            centroids[k] = np.mean(X[mask], axis=0) if np.sum(mask) > 0 else self.cluster_centers_[k]
        return centroids

    def _compute_inertia(self, X, labels, centroids):
        return sum(np.sum((X[labels == k] - centroids[k]) ** 2)
                   for k in range(self.n_clusters) if np.sum(labels == k) > 0)

    def fit(self, X):
        X = np.asarray(X)
        rng = np.random.RandomState(self.random_state)
        best_inertia, best_centroids, best_labels = np.inf, None, None
        for _ in range(self.n_init):
            centroids = self._initialize_centroids(X, rng)
            for _ in range(self.max_iter):
                old = centroids.copy()
                labels = self._assign_clusters(X, centroids)
                self.cluster_centers_ = centroids
                centroids = self._update_centroids(X, labels)
                if np.allclose(centroids, old):
                    break
            inertia = self._compute_inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia, best_centroids, best_labels = inertia, centroids, labels
        self.cluster_centers_, self.labels_, self.inertia_ = best_centroids, best_labels, best_inertia
        return self

    def predict(self, X):
        return self._assign_clusters(np.asarray(X), self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


class BallTreeNumPy:
    """Brute-force nearest-neighbor with haversine — pure NumPy"""

    def __init__(self, data, metric='haversine', leaf_size=40):
        self.data = np.asarray(data)
        self.metric = metric

    def _haversine(self, points, ref):
        lat1, lon1 = ref
        dlat = points[:, 0] - lat1
        dlon = points[:, 1] - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(points[:, 0]) * np.sin(dlon / 2) ** 2
        return 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def query(self, X, k=1):
        X = np.asarray(X)
        distances = np.zeros((len(X), k))
        indices   = np.zeros((len(X), k), dtype=int)
        for i, pt in enumerate(X):
            dists = self._haversine(self.data, pt)
            idx   = np.argpartition(dists, min(k, len(dists) - 1))[:k]
            idx   = idx[np.argsort(dists[idx])]
            distances[i], indices[i] = dists[idx], idx
        return distances, indices

    def query_radius(self, X, r):
        X = np.asarray(X)
        return [np.where(self._haversine(self.data, pt) <= r)[0] for pt in X]


class StandardScalerNumPy:
    """Standardise features — pure NumPy"""

    def fit(self, X):
        X = np.asarray(X)
        self.mean_  = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SimpleImputerNumPy:
    """Impute missing values — pure NumPy"""

    def __init__(self, strategy='median'):
        self.strategy = strategy

    def fit(self, X):
        X = np.asarray(X)
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            from collections import Counter
            self.statistics_ = np.array([
                Counter(X[:, i][~np.isnan(X[:, i])]).most_common(1)[0][0]
                if len(X[:, i][~np.isnan(X[:, i])]) > 0 else 0
                for i in range(X.shape[1])
            ])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = self.statistics_[i]
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class DecisionTreeRegressorNumPy:
    """Decision Tree Regressor — pure NumPy"""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None, criterion='squared_error'):
        self.max_depth        = max_depth if max_depth is not None else 999
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.random_state      = random_state
        self.criterion         = criterion
        self.tree_ = None
        self.feature_importances_ = None

    def _error(self, y):
        if len(y) == 0:
            return 0
        return np.mean(np.abs(y - np.median(y))) if self.criterion == 'absolute_error' \
               else np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y, feature_indices, indices):
        best_gain, best_feat, best_thresh = -np.inf, None, None
        current_error = self._error(y[indices])
        n = len(indices)
        for feat in feature_indices:
            vals = X[indices, feat]
            uniq = np.unique(vals)
            thresholds = np.percentile(vals, np.linspace(5, 95, 20)) if len(uniq) > 20 else uniq[:-1]
            for thresh in thresholds:
                lm = vals <= thresh
                nl, nr = np.sum(lm), n - np.sum(lm)
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gain = current_error - (nl * self._error(y[indices][lm]) +
                                        nr * self._error(y[indices][~lm])) / n
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh

    def _build(self, X, y, indices, depth=0):
        leaf_val = np.median(y[indices]) if self.criterion == 'absolute_error' else np.mean(y[indices])
        if depth >= self.max_depth or len(indices) < self.min_samples_split or np.std(y[indices]) < 1e-7:
            return {'type': 'leaf', 'value': leaf_val, 'n_samples': len(indices)}
        n_feat = X.shape[1]
        if self.max_features == 'sqrt':
            feat_idx = self.rng.choice(n_feat, max(1, int(np.sqrt(n_feat))), replace=False)
        elif isinstance(self.max_features, int):
            feat_idx = self.rng.choice(n_feat, min(self.max_features, n_feat), replace=False)
        else:
            feat_idx = np.arange(n_feat)
        feat, thresh = self._best_split(X, y, feat_idx, indices)
        if feat is None:
            return {'type': 'leaf', 'value': leaf_val, 'n_samples': len(indices)}
        lm = X[indices, feat] <= thresh
        return {'type': 'node', 'feature': feat, 'threshold': thresh, 'n_samples': len(indices),
                'left':  self._build(X, y, indices[lm],  depth + 1),
                'right': self._build(X, y, indices[~lm], depth + 1)}

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.rng = np.random.RandomState(self.random_state)
        self.n_features_ = X.shape[1]
        self.tree_ = self._build(X, y, np.arange(len(X)))
        imp = np.zeros(self.n_features_)
        def _traverse(node):
            if node['type'] == 'leaf': return
            imp[node['feature']] += node['n_samples']
            _traverse(node['left']); _traverse(node['right'])
        _traverse(self.tree_)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.zeros(len(X))
        idx   = np.arange(len(X))
        def _traverse(node, idx):
            if not len(idx): return
            if node['type'] == 'leaf':
                preds[idx] = node['value']; return
            lm = X[idx, node['feature']] <= node['threshold']
            _traverse(node['left'],  idx[lm])
            _traverse(node['right'], idx[~lm])
        _traverse(self.tree_, idx)
        return preds


class RandomForestRegressorNumPy:
    """Random Forest Regressor — pure NumPy"""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None,
                 n_jobs=None, criterion='squared_error'):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.random_state      = random_state
        self.criterion         = criterion
        self.trees_            = []
        self.feature_importances_ = None

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng  = np.random.RandomState(self.random_state)
        self.trees_ = []
        print(f"  Training {self.n_estimators} trees...")
        for i in range(self.n_estimators):
            if (i + 1) % 10 == 0:
                print(f"    Tree {i + 1}/{self.n_estimators}")
            idx = rng.choice(len(X), len(X), replace=True)
            tree = DecisionTreeRegressorNumPy(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                criterion=self.criterion, random_state=rng.randint(0, 100000),
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        print("  Training complete!")
        return self

    def predict(self, X):
        return np.mean([t.predict(np.asarray(X)) for t in self.trees_], axis=0)


# NumPy metric helpers

def _mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def _mse_np(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def _r2_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Training function

def train_numpy_rf(train_df, test_df, train_agg, test_agg):
    print("\n" + "=" * 60)
    print("5 — Random Forest NumPy (from scratch)")
    print("=" * 60)

    TARGET    = 'price_per_sqrtm'
    DROP_COLS = ['price_per_sqrtm', 'valeur_fonciere', 'date_mutation']

    x_train = train_agg.drop(columns=DROP_COLS)
    y_train = train_agg[TARGET]
    x_test  = test_agg.drop(columns=DROP_COLS)
    y_test  = test_agg[TARGET]

    # Numeric features only (NumPy pipeline doesn't handle categoricals)
    numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"  Numeric features : {len(numeric_features)}")

    X_train_num = x_train[numeric_features].values
    X_test_num  = x_test[numeric_features].values

    # Impute + scale
    imputer = SimpleImputerNumPy(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_num)
    X_test_imputed  = imputer.transform(X_test_num)

    scaler = StandardScalerNumPy()
    X_train_proc = scaler.fit_transform(X_train_imputed)
    X_test_proc  = scaler.transform(X_test_imputed)

    # Baseline
    print("\n  Baseline (n=200, depth=30, min_split=15) ...")
    baseline = RandomForestRegressorNumPy(
        n_estimators=200, max_depth=30, min_samples_leaf=1,
        min_samples_split=15, random_state=42, criterion='absolute_error',
    )
    baseline.fit(X_train_proc, y_train.values)
    y_train_pred = baseline.predict(X_train_proc)
    y_test_pred  = baseline.predict(X_test_proc)
    print(f"\n  Baseline — Train  R²={_r2_np(y_train.values, y_train_pred):.4f}  "
          f"MAE={_mae_np(y_train.values, y_train_pred):.2f}  "
          f"RMSE={np.sqrt(_mse_np(y_train.values, y_train_pred)):.2f} €/m²")
    print(f"  Baseline — Test   R²={_r2_np(y_test.values, y_test_pred):.4f}  "
          f"MAE={_mae_np(y_test.values, y_test_pred):.2f}  "
          f"RMSE={np.sqrt(_mse_np(y_test.values, y_test_pred)):.2f} €/m²")

    # Optimised model (best params from sklearn GridSearch: n=250, depth=25, min_split=25)
    print("\n  Optimisé (n=250, depth=25, min_split=25) ...")
    optimized = RandomForestRegressorNumPy(
        n_estimators=250, max_depth=25, min_samples_leaf=1,
        min_samples_split=25, random_state=42, criterion='absolute_error',
    )
    optimized.fit(X_train_proc, y_train.values)
    y_train_pred_opt = optimized.predict(X_train_proc)
    y_test_pred_opt  = optimized.predict(X_test_proc)
    print(f"\n  Optimisé — Train  R²={_r2_np(y_train.values, y_train_pred_opt):.4f}  "
          f"MAE={_mae_np(y_train.values, y_train_pred_opt):.2f}  "
          f"RMSE={np.sqrt(_mse_np(y_train.values, y_train_pred_opt)):.2f} €/m²")
    print(f"  Optimisé — Test   R²={_r2_np(y_test.values, y_test_pred_opt):.4f}  "
          f"MAE={_mae_np(y_test.values, y_test_pred_opt):.2f}  "
          f"RMSE={np.sqrt(_mse_np(y_test.values, y_test_pred_opt)):.2f} €/m²")

    # Save
    preprocessor_np = {'imputer': imputer, 'scaler': scaler, 'feature_names': numeric_features}
    with open(os.path.join(MODEL_DIR, "model_numpy.pkl"), "wb") as f:
        pickle.dump(optimized, f)
    with open(os.path.join(MODEL_DIR, "preprocessor_numpy.pkl"), "wb") as f:
        pickle.dump(preprocessor_np, f)
    print("\n NumPy RF sauvegardé → Model/model_numpy.pkl + Model/preprocessor_numpy.pkl")


# Helpers
def _print_metrics(label, y_train, y_train_pred, y_test, y_test_pred):
    print(f"\n  {label}")
    print(f"    Train  R²={r2_score(y_train, y_train_pred):.4f}  "
          f"MAE={mean_absolute_error(y_train, y_train_pred):.2f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f} €/m²")
    print(f"    Test   R²={r2_score(y_test, y_test_pred):.4f}  "
          f"MAE={mean_absolute_error(y_test, y_test_pred):.2f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f} €/m²")


# Main
if __name__ == "__main__":
    prepare_data()
    train_df, test_df, train_agg, test_agg, km, station_stats_train = build_features()
    train_catboost(train_df, test_df, train_agg, test_agg, km, station_stats_train)
    train_random_forest(train_df, test_df, train_agg, test_agg)
    train_numpy_rf(train_df, test_df, train_agg, test_agg)

    print("\n" + "=" * 60)
    print(" Tout est prêt !")
    print("  Model/model_cat.pkl              +  Model/preprocessor_cat.pkl")
    print("  Model/model_random_forest.pkl    +  Model/preprocessor_random_forest.pkl")
    print("  Model/model_numpy.pkl            +  Model/preprocessor_numpy.pkl")
    print("=" * 60)
