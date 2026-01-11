import os
import time
import re
import ast
import warnings
import shutil
import glob
import logging
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, KFold, cross_validate,
    GridSearchCV, cross_val_score
)
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import category_encoders as ce
import shap
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer

# Optional Dependencies
try:
    import umap
except ImportError:
    umap = None

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    TabularPredictor = None

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [utils] - %(message)s', datefmt='%H:%M:%S')


# Custom Transformers

class PositionalMultiLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Positional Target Encoder for multi-label features.
    Encodes top-K tags based on global frequency and target mean.
    """
    def __init__(self, col_name: str, separator: str = ', ', smoothing: float = 10.0):
        self.col_name = col_name
        self.separator = separator
        self.smoothing = smoothing
        self.mapping_ = {}
        self.frequencies_ = {}
        self.global_mean_ = 0.0
        self.k_ = 0

    def _clean_tags(self, tags_str):
        if pd.isna(tags_str) or str(tags_str).strip() == '' or str(tags_str) == 'N/A':
            return []
        tags = [t.strip() for t in str(tags_str).split(',') if t.strip() != '']
        tags = [re.sub(r'\s*\([^)]*\)', '', t).strip() for t in tags]
        return list(dict.fromkeys([t for t in tags if t != '']))

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        
        tags_series = X[self.col_name].apply(self._clean_tags)
        
        # Calculate K (mean tags per entry)
        self.k_ = int(round(tags_series.apply(len).mean()))
        if self.k_ < 1: self.k_ = 1
        
        # Global statistics
        temp = pd.DataFrame({'tags': tags_series, 'target': y}).explode('tags')
        temp = temp[temp['tags'].notna() & (temp['tags'] != '')]
        
        self.frequencies_ = temp['tags'].value_counts().to_dict()
        
        # Target Encoding with smoothing
        self.global_mean_ = y.mean()
        stats = temp.groupby('tags')['target'].agg(['count', 'mean'])
        m = self.smoothing
        stats['score'] = (stats['count'] * stats['mean'] + m * self.global_mean_) / (stats['count'] + m)
        self.mapping_ = stats['score'].to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        tags_series = X_out[self.col_name].apply(self._clean_tags)
        
        def _get_positional_scores(tags_list):
            if not tags_list:
                return [-1.0] * self.k_
            
            # Sort by global frequency
            tags_sorted = sorted(tags_list, key=lambda x: (self.frequencies_.get(x, 0), x), reverse=True)
            scores = [self.mapping_.get(t, self.global_mean_) for t in tags_sorted]
            
            if len(scores) >= self.k_:
                return scores[:self.k_]
            else:
                return scores + [-1.0] * (self.k_ - len(scores))

        new_cols_data = tags_series.apply(_get_positional_scores)
        
        for i in range(self.k_):
            X_out[f"{self.col_name}_pos_{i+1}"] = new_cols_data.apply(lambda x: x[i])
            
        return X_out.drop(columns=[self.col_name])


# Data Ingestion & Cleaning

def limpiar_dataset_inicial(ruta_entrada: str, ruta_salida: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(ruta_entrada)
        print(f"[INFO] Initial load: {len(df)} rows")
        
        df = df[df['id'] != 'id']
        df = df.dropna(subset=['mi_valoracion'])
        
        cols_borrar = ['estado', 'tipo', 'fecha_vista'] 
        df = df.drop(columns=cols_borrar, errors='ignore')
        
        print(f"[INFO] Valid rows: {len(df)}")
        df.to_csv(ruta_salida, index=False, encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {ruta_entrada}")
        return None

def enriquecer_con_omdb(df: pd.DataFrame, api_key: str, ruta_salida: str) -> Optional[pd.DataFrame]:
    if 'id_imdb' not in df.columns:
        print("[ERROR] Missing 'id_imdb' column")
        return None

    print(f"\n[INFO] Downloading metadata for {len(df)} movies...")

    def _obtener_detalles(imdb_id):
        url = f"http://www.omdbapi.com/?apikey={api_key}&i={imdb_id}&plot=full&r=json"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                data["id_imdb"] = imdb_id 
                return data
            return {"Response": "False", "id_imdb": imdb_id}
        except:
            return {"Response": "False", "id_imdb": imdb_id}

    detalles = []
    for id_val in tqdm(df["id_imdb"], desc="OMDB API"):
        detalles.append(_obtener_detalles(id_val))
        time.sleep(0.05) 

    detalles_df = pd.DataFrame(detalles)
    resultado = df.merge(detalles_df, on="id_imdb", how="left")
    resultado.to_csv(ruta_salida, index=False)
    print(f"[INFO] Saved to: {ruta_salida}")
    return resultado

def limpieza_post_api(df: pd.DataFrame) -> pd.DataFrame:
    cols_eliminar = [
        'titulo', 'Title', 'Poster', 'imdbID', 'Type', 'DVD',
        'Production', 'Website', 'Response', 'Year'
    ]
    df_limpio = df.drop(columns=[c for c in cols_eliminar if c in df.columns])
    df_limpio = df_limpio.drop_duplicates(subset=['id_imdb'])
    print(f"[INFO] Remaining columns: {len(df_limpio.columns)}")
    return df_limpio


# Feature Engineering

def limpiar_features_numericas(df: pd.DataFrame) -> pd.DataFrame:
    # Vectorized string processing
    df['Runtime'] = pd.to_numeric(df['Runtime'].astype(str).str.replace(' min', '', regex=False), errors='coerce')
    df['BoxOffice'] = pd.to_numeric(
        df['BoxOffice'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False),
        errors='coerce'
    )
    
    if 'imdbVotes' in df.columns:
        df['imdbVotes'] = pd.to_numeric(df['imdbVotes'].astype(str).str.replace(',', '', regex=False), errors='coerce')
        
    df['Released'] = pd.to_datetime(df['Released'], errors='coerce')
    df['Year_Released'] = df['Released'].dt.year
    df['imdbRating'] = pd.to_numeric(df['imdbRating'], errors='coerce')
    df['Metascore'] = pd.to_numeric(df['Metascore'], errors='coerce')
    return df

def extraer_awards(df: pd.DataFrame) -> pd.DataFrame:
    def _parse(texto):
        if pd.isna(texto) or texto == 'N/A': return 0, 0, 0
        texto = str(texto)
        
        oscars = re.search(r'Won (\d+) Oscar', texto)
        wins = re.search(r'(\d+) win', texto)
        noms = re.search(r'(\d+) nomination', texto)
        
        return (
            int(oscars.group(1)) if oscars else 0,
            int(wins.group(1)) if wins else 0,
            int(noms.group(1)) if noms else 0
        )

    awards_data = df['Awards'].apply(_parse)
    df[['Awards_Oscars', 'Awards_Wins', 'Awards_Nominations']] = pd.DataFrame(awards_data.tolist(), index=df.index)
    return df

def extraer_rotten_tomatoes(df: pd.DataFrame) -> pd.DataFrame:
    def _get_rotten(texto_lista):
        try:
            if isinstance(texto_lista, str):
                lista = ast.literal_eval(texto_lista)
                for item in lista:
                    if item.get('Source') == 'Rotten Tomatoes':
                        return int(item['Value'].replace('%', ''))
        except: pass
        return np.nan

    df['RottenTomatoes'] = df['Ratings'].apply(_get_rotten)
    return df

def seleccionar_columnas_finales(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'id', 'id_imdb', 'mi_valoracion', 'Year_Released', 'Runtime', 'Genre',
        'Director', 'Writer', 'Actors', 'Plot', 'Language', 'Country',
        'Metascore', 'imdbRating', 'imdbVotes', 'BoxOffice', 'RottenTomatoes',
        'Awards_Oscars', 'Awards_Wins', 'Awards_Nominations'
    ]
    return df[[c for c in cols if c in df.columns]]

def preparar_multi_hot(df: pd.DataFrame, columna: str, min_freq: float = 0.01) -> pd.DataFrame:
    dummies = df[columna].fillna('').str.get_dummies(sep=', ')
    freq = dummies.mean()
    return dummies[freq[freq >= min_freq].index]

def generar_embeddings_plot(df: pd.DataFrame, modelo_nombre: str = "all-mpnet-base-v2") -> pd.DataFrame:
    print(f"\n[NLP] Loading model: {modelo_nombre}...")
    model_st = SentenceTransformer(modelo_nombre)
    
    print(f"[NLP] Generating embeddings for {len(df)} plots...")
    plots = df['Plot'].fillna("").tolist()
    embeddings = model_st.encode(plots, show_progress_bar=True)
    
    cols = [f'plot_{i}' for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=cols, index=df.index)

def guardar_dataset_completo(df_base: pd.DataFrame, df_embeddings: pd.DataFrame, ruta_salida: str) -> pd.DataFrame:
    df_full = pd.concat([df_base, df_embeddings], axis=1)
    df_full.to_csv(ruta_salida, index=False)
    print(f"[INFO] Full dataset saved: {ruta_salida}")
    return df_full


# Evaluation & Modeling Utils

def calcular_metrica_cine(mu_r2, mu_mse, std_r2, std_mse, epsilon=1e-8):
    """
    Calculates combined metric penalizing high variance and MSE.
    """
    def _escalar(val):
        count = 0
        v = abs(val)
        if v > 1:
            while v > 1:
                v /= 10
                count += 1
        return v, count

    s_std_r2, n1 = _escalar(std_r2)
    s_std_mse, n2 = _escalar(std_mse)
    s_mu_mse, n3 = _escalar(mu_mse)

    denominador = (0.5 * (s_std_r2 * s_std_mse) * s_mu_mse + epsilon)
    metric = abs(mu_r2) / denominador

    total_power = n1 + n2 + n3
    metric = metric * (10**total_power)

    return metric * -1 if mu_r2 < 0 else metric

def evaluar_subset(X: pd.DataFrame, y: pd.Series, nombre_subset: str) -> pd.DataFrame:
    modelos = {
        'Lasso (L1)': Lasso(alpha=0.1, random_state=42),
        'Ridge (L2)': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        'CatBoost': CatBoostRegressor(verbose=0, random_state=42, allow_writing_files=False),
        'SVR': SVR(kernel='rbf')
    }
    resultados = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\nEvaluating Subset: {nombre_subset} ({X.shape[1]} features)")

    for name, model in tqdm(modelos.items(), desc=f"Models {nombre_subset}"):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        scores = cross_validate(pipe, X, y, cv=kf, scoring=scoring)
        
        resultados.append({
            'Subset': nombre_subset, 'Modelo': name,
            'R2': scores['test_r2'].mean(),
            'MSE': -scores['test_neg_mean_squared_error'].mean(),
            'MAE': -scores['test_neg_mean_absolute_error'].mean()
        })
    
    df_res = pd.DataFrame(resultados)
    avg = df_res[['R2', 'MSE', 'MAE']].mean()
    avg['Subset'] = nombre_subset; avg['Modelo'] = 'AVERAGE'
    return pd.concat([df_res, pd.DataFrame([avg])], ignore_index=True)

def evaluar_subset_global_target(df: pd.DataFrame, y: pd.Series, col_name: str) -> pd.DataFrame:
    modelos = {
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RF': RandomForestRegressor(n_estimators=50, random_state=42),
        'CatB': CatBoostRegressor(verbose=0, random_state=42, allow_writing_files=False),
        'SVR': SVR()
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    resultados = []
    X_df = df[[col_name]].copy()
    
    print(f"\nEvaluating {col_name} (Global Target Encoding)")

    for name, model in tqdm(modelos.items(), desc=f"Models {col_name}"):
        r2s, mses, maes = [], [], []
        
        for train_idx, val_idx in kf.split(X_df, y):
            enc = PositionalMultiLabelEncoder(col_name=col_name, smoothing=10)
            enc.fit(X_df.iloc[train_idx], y.iloc[train_idx])
            
            X_tr_enc = enc.transform(X_df.iloc[train_idx])
            X_val_enc = enc.transform(X_df.iloc[val_idx])
            
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            pipe.fit(X_tr_enc, y.iloc[train_idx])
            preds = pipe.predict(X_val_enc)
            
            r2s.append(r2_score(y.iloc[val_idx], preds))
            mses.append(mean_squared_error(y.iloc[val_idx], preds))
            maes.append(mean_absolute_error(y.iloc[val_idx], preds))
            
        resultados.append({
            'Subset': col_name, 'Modelo': name, 
            'R2': np.mean(r2s), 'MSE': np.mean(mses), 'MAE': np.mean(maes)
        })
    
    df_res = pd.DataFrame(resultados)
    avg = df_res[['R2', 'MSE', 'MAE']].mean()
    avg['Subset'] = col_name; avg['Modelo'] = 'AVERAGE'
    return pd.concat([df_res, pd.DataFrame([avg])], ignore_index=True)

def comparativa_final_encoding(X_base, y, cols_cat, seeds=[0, 1, 2, 3, 4]):
    res = []
    param_grid = {'model__alpha': np.logspace(-2, 7, 100)}
    print(f"[INFO] Encoding Comparison ({len(seeds)} seeds)")

    for seed in tqdm(seeds, desc="Processing Seeds"):
        X_tr, X_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.2, random_state=seed)
        
        # A. MULTI-HOT
        dummies_tr = [X_tr.select_dtypes(include='number')]
        dummies_te = [X_te.select_dtypes(include='number')]
        
        for c in cols_cat:
            d_tr = X_tr[c].str.get_dummies(sep=', ').add_prefix(f"{c}_")
            d_te = X_te[c].str.get_dummies(sep=', ').add_prefix(f"{c}_").reindex(columns=d_tr.columns, fill_value=-1)
            dummies_tr.append(d_tr)
            dummies_te.append(d_te)
        
        X_tr_mh = pd.concat(dummies_tr, axis=1)
        X_te_mh = pd.concat(dummies_te, axis=1)
        
        for m_name in ['Lasso', 'Ridge']:
            model = Lasso(max_iter=5000) if m_name=='Lasso' else Ridge()
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=-1)), 
                ('scaler', StandardScaler()), 
                ('model', model)
            ])
            gs = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            gs.fit(X_tr_mh, y_tr)
            preds = gs.predict(X_te_mh)
            res.append({
                'Seed': seed, 'Encoding': 'Multi-Hot', 'Modelo': m_name, 
                'R2': r2_score(y_te, preds), 'MSE': mean_squared_error(y_te, preds),
                'MAE': mean_absolute_error(y_te, preds), 'Corr': np.corrcoef(y_te, preds)[0,1]
            })

        # B. GLOBAL TARGET
        X_tr_te = X_tr.copy()
        X_te_te = X_te.copy()
        
        for c in cols_cat:
            enc = PositionalMultiLabelEncoder(col_name=c, smoothing=5.0)
            enc.fit(X_tr, y_tr)
            X_tr_te = enc.transform(X_tr_te) 
            X_te_te = enc.transform(X_te_te)
            
        for m_name in ['Lasso', 'Ridge']:
            model = Lasso(max_iter=5000) if m_name=='Lasso' else Ridge()
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=-1)), 
                ('scaler', StandardScaler()), 
                ('model', model)
            ])
            gs = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            gs.fit(X_tr_te, y_tr)
            preds = gs.predict(X_te_te)
            res.append({
                'Seed': seed, 'Encoding': 'Global-Target', 'Modelo': m_name, 
                'R2': r2_score(y_te, preds), 'MSE': mean_squared_error(y_te, preds),
                'MAE': mean_absolute_error(y_te, preds), 'Corr': np.corrcoef(y_te, preds)[0,1]
            })
            
    return pd.DataFrame(res)


# Visualization & Embeddings

def visualizar_plot_2d(X_plot: pd.DataFrame, y: pd.Series):
    def _get_color_category(nota):
        if nota <= 6.0: return 'Low (<=6)'
        elif nota == 6.5: return 'Mid (6.5)'
        else: return 'High (>=7)'

    categorias = y.apply(_get_color_category)

    print("[INFO] Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_plot)

    X_umap = None
    if umap is not None:
        try:
            print("[INFO] Computing UMAP...")
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            X_umap = reducer.fit_transform(X_plot)
        except Exception:
            print("[WARN] UMAP failed.")
    else:
        print("[WARN] UMAP not installed. Skipping.")

    n_plots = 2 if X_umap is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(16, 7))
    if n_plots == 1: axes = [axes]

    palette = {'Low (<=6)': 'red', 'Mid (6.5)': '#FFD700', 'High (>=7)': 'green'}

    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=categorias, palette=palette, ax=axes[0], alpha=0.7)
    axes[0].set_title('PCA of Plot')

    if X_umap is not None:
        sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=categorias, palette=palette, ax=axes[1], alpha=0.7)
        axes[1].set_title('UMAP of Plot')
        
    plt.tight_layout()
    plt.show()

def evaluar_knn_plot_robusto(X, y, seeds=[0]):
    resultados = []
    k_range = np.arange(1, 101, 2)
    param_grid = {'knn__n_neighbors': k_range}
    
    print(f"\n[INFO] Starting Robust KNN ({len(seeds)} seeds)")

    for seed in tqdm(seeds, desc="KNN Seeds"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        pipelines = {
            'Euclidean': Pipeline([('knn', KNeighborsRegressor(metric='euclidean'))]),
            'Cosine (L2)': Pipeline([('norm', Normalizer(norm='l2')), ('knn', KNeighborsRegressor(metric='euclidean'))])
        }
        
        for metric_name, pipe in pipelines.items():
            gs = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
            
            resultados.append({
                'Seed': seed, 'Metric': metric_name,
                'Best_K': gs.best_params_['knn__n_neighbors'],
                'R2': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'Corr': np.corrcoef(y_test, y_pred)[0,1]
            })
    return pd.DataFrame(resultados)

def evaluar_pca_knn_robusto(X, y, seeds=[0]):
    resultados = []
    variance_range = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    k_range = np.arange(1, 51, 5)
    param_grid = {'pca__n_components': variance_range, 'knn__n_neighbors': k_range}

    print(f"\n[INFO] Starting Robust PCA+KNN ({len(seeds)} seeds)")

    for seed in tqdm(seeds, desc="PCA+KNN Seeds"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        pipelines = {
            'Euclidean': Pipeline([
                ('pca', PCA(random_state=42)),
                ('knn', KNeighborsRegressor(metric='euclidean'))
            ]),
            'Cosine (L2)': Pipeline([
                ('norm', Normalizer(norm='l2')),
                ('pca', PCA(random_state=42)),
                ('knn', KNeighborsRegressor(metric='euclidean'))
            ])
        }
        
        for metric_name, pipe in pipelines.items():
            gs = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
            n_dims = best_model.named_steps['pca'].n_components_
            
            resultados.append({
                'Seed': seed, 'Metric': metric_name,
                'Best_Var': gs.best_params_['pca__n_components'],
                'Real_Dims': n_dims,
                'Best_K': gs.best_params_['knn__n_neighbors'],
                'R2': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'Corr': np.corrcoef(y_test, y_pred)[0,1]
            })
    return pd.DataFrame(resultados)


# User Data Integration

def cargar_datos_usuarios_recursivo(ruta_raiz: str, df_base: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if 'id' not in df_base.columns:
        print("[ERROR] Base dataframe requires 'id' column")
        return df_base, []

    print(f"\n[INFO] Loading users from '{ruta_raiz}'...")
    df_merged = df_base.copy()
    cols_nuevas = []

    if not os.path.exists(ruta_raiz):
        print(f"[ERROR] Path does not exist: {ruta_raiz}")
        return df_base, []

    archivos_candidatos = []
    
    # 1. Folders
    for d in os.listdir(ruta_raiz):
        p = os.path.join(ruta_raiz, d)
        if os.path.isdir(p):
            csvs = glob.glob(os.path.join(p, "*.csv"))
            if csvs: archivos_candidatos.append((csvs[0], d))

    # 2. Files
    directos = glob.glob(os.path.join(ruta_raiz, "*_valoraciones.csv"))
    for f in directos:
        user_id = os.path.basename(f).replace('_valoraciones.csv', '')
        archivos_candidatos.append((f, user_id))

    for ruta_f, user_id in archivos_candidatos:
        nombre_col = f'user_{user_id}'
        if nombre_col in cols_nuevas: continue
        
        try:
            df_user = pd.read_csv(ruta_f)
            if 'tmdb_id' in df_user.columns and 'user_rating' in df_user.columns:
                df_user = df_user[['tmdb_id', 'user_rating']]
                df_user.columns = ['tmdb_id', nombre_col]
                df_user[nombre_col] = pd.to_numeric(df_user[nombre_col], errors='coerce')
                
                df_merged = df_merged.merge(df_user, left_on='id', right_on='tmdb_id', how='left')
                if 'tmdb_id' in df_merged.columns:
                    df_merged = df_merged.drop(columns=['tmdb_id'])
                
                cols_nuevas.append(nombre_col)
        except Exception as e:
            print(f"   [WARN] Error processing {user_id}: {e}")
            
    print(f"[INFO] Integrated {len(cols_nuevas)} users.")
    df_merged[cols_nuevas] = df_merged[cols_nuevas].fillna(-1)
    return df_merged, cols_nuevas

def cargar_referencia_usuarios(ruta_raiz: str) -> Tuple[pd.DataFrame, List[str]]:
    print(f"[INFO] Indexing user database from '{ruta_raiz}'...")
    data_map = {}
    cols_users = []

    archivos_candidatos = []
    if os.path.exists(ruta_raiz):
        for d in os.listdir(ruta_raiz):
            p = os.path.join(ruta_raiz, d)
            if os.path.isdir(p):
                csvs = glob.glob(os.path.join(p, "*.csv"))
                if csvs: archivos_candidatos.append((csvs[0], d))
        for f in glob.glob(os.path.join(ruta_raiz, "*_valoraciones.csv")):
            archivos_candidatos.append((f, os.path.basename(f).replace('_valoraciones.csv', '')))

    for ruta_f, user_id in archivos_candidatos:
        try:
            df = pd.read_csv(ruta_f)
            if 'tmdb_id' not in df.columns or 'user_rating' not in df.columns: continue
            
            user_col = f'user_{user_id}'
            cols_users.append(user_col)
            
            for _, row in df.iterrows():
                tmdb_id = str(row['tmdb_id']) 
                nota = row['user_rating']
                
                if tmdb_id not in data_map:
                    data_map[tmdb_id] = {'tmdb_id': tmdb_id}
                data_map[tmdb_id][user_col] = nota
        except: pass
        
    if not data_map: return pd.DataFrame(), []
    df_master = pd.DataFrame.from_dict(data_map, orient='index').fillna(-1)
    return df_master, cols_users


# Optimization (Optuna & AutoGluon)

def optimizar_arquitectura_unica(X_dict, y, n_trials=50, n_seeds_inner=2, ruta_guardado='resultados_optuna_unicos.csv'):
    study_results = []
    seen_configs = set()
    alphas_ridge = np.logspace(-2, 7, 100)
    
    print(f"\n[OPTUNA] Optimization ({n_trials} Trials x {n_seeds_inner} Seeds)")

    def objective(trial):
        use_num = trial.suggest_categorical('use_num', [True, False])
        use_users = trial.suggest_categorical('use_users', [True, False])
        cats_activos = []
        for cat in ['Genre', 'Director', 'Writer', 'Actors', 'Language', 'Country']:
            if trial.suggest_categorical(f'use_{cat}', [True, False]):
                cats_activos.append(cat)
        
        smoothing = trial.suggest_int('smoothing', 1, 5)
        model_type = trial.suggest_categorical('model_type', ['Lasso', 'Ridge'])
        
        cats_tuple = tuple(sorted(cats_activos))
        config_signature = (use_num, use_users, cats_tuple, model_type, smoothing)
        
        if config_signature in seen_configs: raise optuna.TrialPruned()
        if not use_num and not use_users and not cats_activos: raise optuna.TrialPruned()
        seen_configs.add(config_signature)
        
        r2_list, mse_list, mae_list, corr_list, alpha_list = [], [], [], [], []
        seeds = np.arange(50, 50 + n_seeds_inner)
        
        for seed in seeds:
            idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
            y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
            
            X_tr_parts, X_te_parts = [], []
            if use_num:
                X_tr_parts.append(X_dict['NUM'].iloc[idx_train])
                X_te_parts.append(X_dict['NUM'].iloc[idx_test])
            if use_users:
                X_tr_parts.append(X_dict['USERS'].iloc[idx_train])
                X_te_parts.append(X_dict['USERS'].iloc[idx_test])
                
            for cat in cats_activos:
                enc = PositionalMultiLabelEncoder(col_name=cat, smoothing=float(smoothing))
                enc.fit(X_dict[cat].iloc[idx_train], y_train)
                X_tr_parts.append(enc.transform(X_dict[cat].iloc[idx_train]))
                X_te_parts.append(enc.transform(X_dict[cat].iloc[idx_test]))
                
            X_tr = pd.concat(X_tr_parts, axis=1)
            X_te = pd.concat(X_te_parts, axis=1)
            
            pre = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value=-1)), ('scl', StandardScaler())])
            X_tr_p = pre.fit_transform(X_tr)
            X_te_p = pre.transform(X_te)
            
            if model_type == 'Lasso':
                model = LassoCV(cv=5, random_state=42, max_iter=2000, alphas=None).fit(X_tr_p, y_train)
            else:
                model = RidgeCV(cv=5, alphas=alphas_ridge).fit(X_tr_p, y_train)
                
            preds = model.predict(X_te_p)
            r2_list.append(r2_score(y_test, preds))
            mse_list.append(mean_squared_error(y_test, preds))
            mae_list.append(mean_absolute_error(y_test, preds))
            corr_list.append(np.corrcoef(y_test, preds)[0,1])
            alpha_list.append(model.alpha_)
            
        mu_r2, std_r2 = np.mean(r2_list), np.std(r2_list)
        mu_mse, std_mse = np.mean(mse_list), np.std(mse_list)

        combined_metric = calcular_metrica_cine(mu_r2, mu_mse, std_r2, std_mse)
        
        study_results.append({
            'Trial': trial.number, 'Metric_Comb': combined_metric,
            'R2_Mean': mu_r2, 'R2_Std': std_r2,
            'MSE_Mean': mu_mse, 'MSE_Std': std_mse,
            'MAE_Mean': np.mean(mae_list), 'Corr_Mean': np.mean(corr_list),
            'Model': model_type, 'Avg_Alpha': np.mean(alpha_list),
            'Smooth': smoothing, 'Use_Num': use_num, 'Use_Users': use_users,
            'Cats_Active': list(cats_activos)
        })
        return combined_metric

    sampler = TPESampler(n_startup_trials=int(n_trials * 0.25), seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    pbar = tqdm(total=n_trials, desc="Trials")
    trials_done = 0
    while trials_done < n_trials:
        study.optimize(objective, n_trials=1)
        if study.trials[-1].state == optuna.trial.TrialState.COMPLETE:
            trials_done += 1
            pbar.update(1)
    pbar.close()

    df_res = pd.DataFrame(study_results).sort_values('Metric_Comb', ascending=False)
    df_res['Cats_Active'] = df_res['Cats_Active'].apply(lambda x: '+'.join(x) if x else 'None')
    df_res.to_csv(ruta_guardado, index=False)
    print(f"[INFO] Results saved to: {ruta_guardado}")
    return df_res

def optimizar_arquitectura_unica_rf(X_dict, y, n_trials=50, n_seeds_inner=2, ruta_guardado='resultados_optuna_rf.csv'):
    study_results = []
    seen_configs = set()
    print(f"\n[OPTUNA] Architecture + RF ({n_trials} Trials)")

    def objective(trial):
        use_num = trial.suggest_categorical('use_num', [True, False])
        use_users = trial.suggest_categorical('use_users', [True, False])
        cats_activos = []
        for cat in ['Genre', 'Director', 'Writer', 'Actors', 'Language', 'Country']:
            if trial.suggest_categorical(f'use_{cat}', [True, False]): cats_activos.append(cat)
        
        if not use_num and not use_users and not cats_activos: raise optuna.TrialPruned()
        
        smoothing = trial.suggest_int('smoothing', 1, 5)
        n_est = trial.suggest_int('rf_n_estimators', 50, 300)
        max_depth = trial.suggest_int('rf_max_depth', 3, 20)
        min_samples = trial.suggest_int('rf_min_samples', 2, 10)
        max_feat = trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', 1.0])
        
        config_signature = (use_num, use_users, tuple(sorted(cats_activos)), smoothing, n_est//50, max_depth//2, max_feat)
        if config_signature in seen_configs: raise optuna.TrialPruned()
        seen_configs.add(config_signature)
        
        r2_list, mse_list, mae_list, corr_list = [], [], [], []
        seeds = np.arange(123, 123 + n_seeds_inner)
        
        for seed in seeds:
            idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
            y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
            
            parts_tr, parts_te = [] , []
            if use_num:
                parts_tr.append(X_dict['NUM'].iloc[idx_train]); parts_te.append(X_dict['NUM'].iloc[idx_test])
            if use_users:
                parts_tr.append(X_dict['USERS'].iloc[idx_train]); parts_te.append(X_dict['USERS'].iloc[idx_test])
            for cat in cats_activos:
                enc = PositionalMultiLabelEncoder(col_name=cat, smoothing=float(smoothing))
                enc.fit(X_dict[cat].iloc[idx_train], y_train)
                parts_tr.append(enc.transform(X_dict[cat].iloc[idx_train]))
                parts_te.append(enc.transform(X_dict[cat].iloc[idx_test]))
                
            X_tr = pd.concat(parts_tr, axis=1)
            X_te = pd.concat(parts_te, axis=1)
            
            model = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, min_samples_split=min_samples, max_features=max_feat, random_state=42, n_jobs=-1)
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('model', model)])
            
            pipe.fit(X_tr, y_train)
            preds = pipe.predict(X_te)
            
            r2_list.append(r2_score(y_test, preds))
            mse_list.append(mean_squared_error(y_test, preds))
            mae_list.append(mean_absolute_error(y_test, preds))
            corr_list.append(np.corrcoef(y_test, preds)[0,1])
            
        mu_r2, std_r2 = np.mean(r2_list), np.std(r2_list)
        mu_mse, std_mse = np.mean(mse_list), np.std(mse_list)

        metric = calcular_metrica_cine(mu_r2, mu_mse, std_r2, std_mse)
        
        study_results.append({
            'Trial': trial.number, 'Metric_Comb': metric,
            'R2_Mean': mu_r2, 'R2_Std': std_r2, 'MSE_Mean': mu_mse, 'MSE_Std': std_mse,
            'MAE_Mean': np.mean(mae_list), 'Corr_Mean': np.mean(corr_list),
            'Model': 'RandomForest', 'Params_RF': str({'n':n_est, 'd':max_depth}),
            'Smooth': smoothing, 'Use_Num': use_num, 'Use_Users': use_users,
            'Cats_Active': list(cats_activos)
        })
        return metric

    sampler = TPESampler(n_startup_trials=int(n_trials * 0.25), seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    pbar = tqdm(total=n_trials, desc="Trials RF")
    trials_done = 0
    while trials_done < n_trials:
        study.optimize(objective, n_trials=1)
        if study.trials[-1].state == optuna.trial.TrialState.COMPLETE:
            trials_done += 1
            pbar.update(1)
    pbar.close()

    df_res = pd.DataFrame(study_results).sort_values('Metric_Comb', ascending=False)
    df_res['Cats_Active'] = df_res['Cats_Active'].apply(lambda x: '+'.join(x) if x else 'None')
    df_res.to_csv(ruta_guardado, index=False)
    return df_res

def optimizar_arquitectura_autogluon(X_dict, y, n_trials=20, n_seeds=10, time_limit=30, ruta_guardado='resultados_optuna_ag.csv'):
    if TabularPredictor is None:
        print("[ERROR] AutoGluon not installed.")
        return pd.DataFrame()

    study_results = []
    seen_configs = set()
    print(f"\n[AUTOGLUON] Optimization ({n_trials} Trials, {time_limit}s/seed)")

    def objective(trial):
        use_num = trial.suggest_categorical('use_num', [True, False])
        use_users = trial.suggest_categorical('use_users', [True, False])
        cats_activos = []
        for cat in ['Genre', 'Director', 'Writer', 'Actors', 'Language', 'Country']:
            if trial.suggest_categorical(f'use_{cat}', [True, False]): 
                cats_activos.append(cat)
        
        cats_tuple = tuple(sorted(cats_activos))
        config_signature = (use_num, use_users, cats_tuple)
        
        if config_signature in seen_configs or (not use_num and not use_users and not cats_activos):
            raise optuna.TrialPruned()
        seen_configs.add(config_signature)
        
        parts = []
        if use_num: parts.append(X_dict['NUM'])
        if use_users: parts.append(X_dict['USERS'])
        for cat in cats_activos: parts.append(X_dict[cat])
        X_curr = pd.concat(parts, axis=1)
        
        r2_list, mse_list, mae_list, corr_list = [], [], [], []
        seeds = np.arange(654, 654 + n_seeds)
        
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(X_curr, y, test_size=0.2, random_state=seed)
            train_data = X_train.copy()
            train_data['target'] = y_train
            path_tmp = f"ag_optuna_t{trial.number}_s{seed}"
            
            try:
                predictor = TabularPredictor(label='target', path=path_tmp, verbosity=0).fit(
                    train_data, presets='medium_quality', time_limit=time_limit, 
                    excluded_model_types=['TextPredictor']
                )
                preds = predictor.predict(X_test)
                r2_list.append(r2_score(y_test, preds))
                mse_list.append(mean_squared_error(y_test, preds))
                mae_list.append(mean_absolute_error(y_test, preds))
                corr_list.append(np.corrcoef(y_test, preds)[0,1])
            except Exception as e:
                return -9999.0
            finally:
                try: shutil.rmtree(path_tmp)
                except: pass
        
        mu_r2, std_r2 = np.mean(r2_list), np.std(r2_list)
        mu_mse, std_mse = np.mean(mse_list), np.std(mse_list)
        
        metric = calcular_metrica_cine(mu_r2, mu_mse, std_r2, std_mse)
        
        study_results.append({
            'Trial': trial.number, 
            'Metric_Comb': metric,
            'R2_Mean': mu_r2, 
            'R2_Std': std_r2, 
            'MSE_Mean': mu_mse, 
            'MSE_Std': std_mse,
            'MAE_Mean': np.mean(mae_list), 
            'Corr_Mean': np.mean(corr_list),
            'Use_Num': use_num, 
            'Use_Users': use_users,
            'Cats_Active': '+'.join(cats_activos) if cats_activos else 'None'
        })
        return metric

    sampler = TPESampler(n_startup_trials=int(n_trials * 0.25), seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    pbar = tqdm(total=n_trials, desc="Trials AG")
    trials_done = 0
    while trials_done < n_trials:
        study.optimize(objective, n_trials=1)
        if study.trials[-1].state == optuna.trial.TrialState.COMPLETE:
            trials_done += 1
            pbar.update(1)
    pbar.close()

    df_res = pd.DataFrame(study_results).sort_values('Metric_Comb', ascending=False)
    df_res.to_csv(ruta_guardado, index=False)
    print(f"[INFO] AutoGluon results saved: {ruta_guardado}")
    return df_res


# Final Training & SHAP

def entrenar_rf_final_optimizado(X_final, y, cat_cols, n_trials=50):
    print(f"\n[INFO] Hyperparameter tuning ({n_trials} trials)...")
    col_objetivo = cat_cols[0] if isinstance(cat_cols, list) else cat_cols

    def objective(trial):
        n_est = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 30)
        min_samples = trial.suggest_int('min_samples_split', 2, 10)
        max_feat = trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0])
        smoothing = trial.suggest_int('smoothing', 1, 5) 

        pipe = Pipeline([
            ('pos_target_enc', PositionalMultiLabelEncoder(col_name=col_objetivo, smoothing=smoothing)),
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, 
                                          min_samples_split=min_samples, max_features=max_feat,
                                          random_state=42, n_jobs=-1))
        ])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(pipe, X_final, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1).mean()

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    print(f"[INFO] Best Params: {best}")

    maestro_pipe = Pipeline([
        ('pos_target_enc', PositionalMultiLabelEncoder(col_name=col_objetivo, smoothing=best['smoothing'])),
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=best['n_estimators'], max_depth=best['max_depth'], 
                                      min_samples_split=best['min_samples_split'], max_features=best['max_features'],
                                      random_state=42, n_jobs=-1))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores, mse_scores, mae_scores, corr_scores = [], [], [], []
    y_test_plot, y_pred_plot = None, None

    print("\n[INFO] Final Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_final, y)):
        X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        maestro_pipe.fit(X_train, y_train)
        y_pred = maestro_pipe.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        corr_scores.append(np.corrcoef(y_test, y_pred)[0,1])
        if fold == 0: y_test_plot, y_pred_plot = y_test, y_pred

    print("\n[METRICS] KFold CV Results:")
    print(f"R2: {np.mean(r2_scores):.4f} | MSE: {np.mean(mse_scores):.4f} | MAE: {np.mean(mae_scores):.4f} | Corr: {np.mean(corr_scores):.4f}")

    print("\n[INFO] Retraining on full dataset...")
    maestro_pipe.fit(X_final, y) 
    joblib.dump(maestro_pipe, 'modelo_cine_rf_final.pkl')

    # Plot Real vs Predicted
    plt.figure(figsize=(8, 6))
    br = BayesianRidge().fit(y_test_plot.values.reshape(-1, 1), y_pred_plot)
    line_x = np.linspace(y_test_plot.min(), y_test_plot.max(), 100)
    plt.scatter(y_test_plot, y_pred_plot, alpha=0.5, color='blue', label='Test Data')
    plt.plot(line_x, br.predict(line_x.reshape(-1, 1)), color='red', label='Trend')
    plt.plot([y_test_plot.min(), y_test_plot.max()], [y_test_plot.min(), y_test_plot.max()], 'k--', alpha=0.5, label='Perfect Fit')
    plt.title("Prediction vs Reality (Final Model)"); plt.legend(); plt.show()

    # SHAP Analysis
    print("\n[INFO] Generating SHAP values...")
    try:
        model_step = maestro_pipe.named_steps['model']
        preprocessor = Pipeline(maestro_pipe.steps[:-1])
        X_trans = preprocessor.transform(X_final)
        
        feature_names = [c for c in X_final.columns if c != col_objetivo]
        k_final = preprocessor.named_steps['pos_target_enc'].k_
        for i in range(k_final): feature_names.append(f"{col_objetivo}_pos_{i+1}")
        
        explainer = shap.TreeExplainer(model_step)
        shap_values = explainer.shap_values(X_trans)
        
        plt.figure(); plt.title("SHAP Global Summary")
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, max_display=15, show=False); plt.show()
        
        # Local SHAP
        notas_objetivo = [5.0, 6.5, 8.0]
        print(f"\n[INFO] Local analysis for target scores: {notas_objetivo}")
        
        for nota in notas_objetivo:
            indices = np.where(y == nota)[0]
            if len(indices) > 0:
                idx = indices[0]
                plt.figure()
                shap_exp = shap.Explanation(
                    values=shap_values[idx],
                    base_values=explainer.expected_value,
                    data=X_trans[idx],
                    feature_names=feature_names
                )
                print(f"-> Waterfall plot for score: {nota}")
                shap.plots.waterfall(shap_exp, show=True)
            else:
                print(f"-> [WARN] No movie found with exact score {nota}.")

    except Exception as e: 
        print(f"[WARN] SHAP failed: {e}")
        
    return maestro_pipe


# Deployment

class CinePredictor:
    def __init__(self,
                 model_path: str = 'modelo_cine_final.pkl',
                 users_path: str = 'data',
                 api_key: str = None,
                 mapping_path: str = 'biblioteca_peliculas_limpia.csv'):
        
        self.api_key = api_key
        self.es_autogluon = False
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] Model file not found: {model_path}")

        obj = joblib.load(model_path)
        
        if isinstance(obj, dict) and obj.get('tipo') == 'AutoGluon':
            from autogluon.tabular import TabularPredictor
            self.model = TabularPredictor.load(obj['path'])
            self.es_autogluon = True
            self.feature_names = self.model.feature_metadata.get_features()
            print("[INFO] CinePredictor: Loaded AutoGluon model.")
        else:
            self.model = obj
            self.es_autogluon = False
            self.feature_names = list(self.model.feature_names_in_)
            print(f"[INFO] CinePredictor: Loaded Sklearn model ({type(self.model.named_steps['model']).__name__}).")

        if users_path and os.path.exists(users_path):
            self.db_users, self.cols_users = cargar_referencia_usuarios(users_path)
        else:
            self.db_users, self.cols_users = None, []

        try:
            df_map = pd.read_csv(mapping_path)
            self.map_imdb_to_tmdb = df_map.set_index('id_imdb')['id'].to_dict()
        except:
            print("[WARN] Failed to load IMDB->TMDB mapping. Social features may fail.")
            self.map_imdb_to_tmdb = {}

    def predecir(self, query: str) -> Union[Dict, str]:
        print(f"\n[QUERY] Searching for: '{query}'")
        param = 'i' if query.startswith('tt') else 't'
        url = f"http://www.omdbapi.com/?apikey={self.api_key}&{param}={query}&plot=full&r=json"
        
        try:
            raw_data = requests.get(url, timeout=5).json()
            if raw_data.get('Response') == 'False': 
                return f"[ERROR] OMDB API: {raw_data.get('Error')}"
        except Exception as e: 
            return f"[ERROR] API Connection: {e}"

        id_imdb = raw_data.get('imdbID')
        titulo = raw_data.get('Title')
        print(f"[MATCH] {titulo} ({id_imdb})")
        
        X_input = pd.DataFrame(index=[0], columns=self.feature_names)
        
        # Social Data Fill
        vistos = 0
        tmdb_id = self.map_imdb_to_tmdb.get(id_imdb)
        cols_user_en_modelo = [c for c in self.feature_names if c.startswith('user_')]
        
        if tmdb_id and self.db_users is not None and str(tmdb_id) in self.db_users.index:
            row_social = self.db_users.loc[str(tmdb_id)]
            for col in cols_user_en_modelo:
                val = row_social.get(col, -1.0)
                X_input.at[0, col] = float(val)
                if val > 0: vistos += 1
            print(f"[SOCIAL] {vistos} friends watched this.")
        else:
            for col in cols_user_en_modelo:
                X_input.at[0, col] = -1.0

        # Categorical Data Fill
        for cat in ['Genre', 'Director', 'Writer', 'Actors', 'Language', 'Country']:
            if cat in self.feature_names:
                X_input.at[0, cat] = raw_data.get(cat, "Unknown")

        # Prediction
        try:
            if self.es_autogluon:
                nota = self.model.predict(X_input)[0]
            else:
                nota = self.model.predict(X_input)[0]
            
            return {
                'Title': titulo, 
                'Prediction': round(max(0, min(10, nota)), 2), 
                'Writer': raw_data.get('Writer', 'N/A'),
                'Friends_Watched': vistos,
                'ID_IMDB': id_imdb
            }
        except Exception as e: 
            return f"[ERROR] Prediction failed: {e}"
