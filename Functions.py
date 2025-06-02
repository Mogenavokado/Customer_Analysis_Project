"""
Clustering Pipeline
For readability, modularity, and maintainability.
Includes preprocessing, feature engineering, scaling, dimensionality reduction,
clustering, profiling, visualization, and stability analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from collections import Counter
from sklearn.metrics import silhouette_score

# --- Configuration ---
DATE_COLS = [
    'transaction_date', 'last_purchase_date',
    'promotion_start_date', 'promotion_end_date',
    'product_manufacture_date', 'product_expiry_date'
]
CAT_COLS = [
    'gender', 'income_bracket', 'marital_status', 'loyalty_program',
    'education_level', 'occupation', 'app_usage', 'social_media_engagement',
    'promotion_type', 'promotion_channel', 'promotion_effectiveness'
]
FREQ_MAP = {'Daily': 365, 'Weekly': 52, 'Monthly': 12, 'Yearly': 1}

# --- Preprocessing ---
def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert specified columns to datetime, coercing errors."""
    df = df.copy()
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df.get(col), errors='coerce')
    return df

# --- Feature Engineering ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric features and customer lifetime proxy."""
    df = df.copy()
    df['net_sales'] = df['total_sales'] - df['total_returned_value']
    df['discount_rate'] = (df['total_discounts_received'] / df['total_sales']).fillna(0)
    df['return_rate'] = (df['total_returned_items'] / df['total_items_purchased']).fillna(0)
    df['online_ratio'] = df['online_purchases'] / df['total_transactions']
    df['in_store_ratio'] = df['in_store_purchases'] / df['total_transactions']
    df['purchases_per_year'] = df['purchase_frequency'].map(FREQ_MAP).fillna(0)
    df['months_since_last'] = df['days_since_last_purchase'] / 30
    df['avg_order_value'] = df['net_sales'] / df['total_transactions']
    df['clv_proxy'] = df['avg_order_value'] * df['purchases_per_year']

    # RFM metrics
    df['recency'] = df['days_since_last_purchase']
    df['frequency'] = df['total_transactions']
    df['monetary'] = df['net_sales']
    return df

# --- Encoding ---
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode configured categorical columns."""
    return pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

# --- Feature Selection ---
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select baseline and one-hot encoded features for clustering."""
    baseline_feats = [
        'customer_id', 'age', 'membership_years',
        'days_since_last_purchase', 'total_transactions', 'total_sales',
        'net_sales', 'discount_rate', 'return_rate',
        'online_ratio', 'in_store_ratio', 'purchases_per_year',
        'months_since_last', 'avg_order_value', 'clv_proxy',

        # RFM metrics
        'recency', 'frequency', 'monetary'
    ]
    encoded_feats = [col for col in df.columns if any(col.startswith(f + '_') for f in CAT_COLS)]
    return df[baseline_feats + encoded_feats]

# --- Outlier Treatment ---
def clip_outliers(df: pd.DataFrame, cols: list, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip numeric columns at given quantiles."""
    df = df.copy()
    mins = df[cols].quantile(lower)
    maxs = df[cols].quantile(upper)
    df[cols] = df[cols].clip(lower=mins, upper=maxs, axis=1)
    return df

# --- Scaling ---
def scale_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Standardize numeric columns."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols])
    return pd.DataFrame(scaled, columns=cols, index=df.index)

# --- Dimensionality Reduction ---
def run_pca(df: pd.DataFrame, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(df)
    print(f"PCA variance ratio: {pca.explained_variance_ratio_}")
    plt.scatter(comps[:, 0], comps[:, 1], s=10)
    plt.title('PCA')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.show()
    return comps


def run_tsne(df: pd.DataFrame, perplexity: int = None) -> np.ndarray:
    if perplexity is None:
        perplexity = min(30, max(5, len(df) // 3))
    emb = TSNE(n_components=2, perplexity=perplexity,
               random_state=42, init='pca', learning_rate='auto').fit_transform(df)
    plt.scatter(emb[:, 0], emb[:, 1], s=10)
    plt.title(f't-SNE (perplexity={perplexity})')
    plt.xlabel('Dim1'); plt.ylabel('Dim2'); plt.show()
    return emb

# --- Clustering & Profiling ---
def cluster_and_profile(df_scaled: pd.DataFrame, df_raw: pd.DataFrame, k: int = 4):
    """Fit KMeans and return cluster labels and aggregated profile."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(df_scaled)
    df_raw = df_raw.copy()
    df_raw['cluster'] = labels
    profile = (
        df_raw.groupby('cluster')
              .agg(count=('customer_id','size'),
                   net_sales=('net_sales','mean'),
                   avg_order_value=('avg_order_value','mean'),
                   clv_proxy=('clv_proxy','mean'),
                   transactions=('total_transactions','mean'),
                   recency_months=('months_since_last','mean'),
                   recency=('recency','mean'),
                   frequency=('frequency','mean'),
                   monetary=('monetary','mean'),
                   rfm_score =('RFM_Score','mean'))
              .reset_index()
    )
    
    # This is optional: Top RFM segments per cluster (for interpretation)
    top_segments = (
        df_raw.groupby('cluster')['RFM_Segment']
        .apply(lambda x: Counter(x).most_common(1)[0][0])
        .reset_index()
        .rename(columns={'RFM_Segment': 'top_rfm_segment'})
    )

    # Merge top RFM segment info into profile
    profile = profile.merge(top_segments, on='cluster')


    return labels, profile

# --- Stability Analysis ---
def bootstrap_stability(df_scaled: pd.DataFrame, labels: np.ndarray, n_runs: int = 10, frac: float = 0.8) -> float:
    """Compute mean Adjusted Rand Index over bootstrap samples."""
    ari_scores = []
    idx = np.arange(len(df_scaled))
    for _ in range(n_runs):
        sample = np.random.choice(idx, int(frac*len(idx)), replace=False)
        km = KMeans(n_clusters=len(np.unique(labels)), random_state=42, n_init=10)
        km.fit(df_scaled.iloc[sample])
        ari_scores.append(adjusted_rand_score(labels[sample], km.labels_))
    mean_ari = float(np.mean(ari_scores))
    print(f"Mean ARI over {n_runs} bootstraps: {mean_ari:.3f}")
    return mean_ari


# --- RFM Scoring ---
def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RFM scores and segments."""
    df = df.copy()
    df['R_Score'] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1])
    df['F_Score'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    df['M_Score'] = pd.qcut(df['monetary'], 5, labels=[1, 2, 3, 4, 5])
    df['RFM_Segment'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
    df['RFM_Score'] = df[['R_Score', 'F_Score', 'M_Score']].astype(int).sum(axis=1)
    return df

def compute_silhouette(X: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Compute and return the average silhouette score for a clustering.
    - X: feature matrix (scaled)
    - labels: cluster assignments
    """
    score = silhouette_score(X, labels)
    print(f"[Silhouette] Avg. score: {score:.3f}")
    return score

# --- Silhouette Analysis ---
def plot_silhouette_vs_k(
    X: pd.DataFrame,
    k_range: range = range(2, 8),
    random_state: int = 42
) -> None:
    """
    Plot average silhouette scores over a range of k for KMeans.
    - X: feature matrix (scaled)
    - k_range: iterable of k values to try
    """
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state).fit(X)
        sil_scores.append(silhouette_score(X, km.labels_))
    plt.figure(figsize=(8,4))
    plt.plot(list(k_range), sil_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Avg. Silhouette Score")
    plt.title("Silhouette Analysis for k")
    plt.tight_layout()
    plt.show()

# --- Visualization ---

# Plot the elbow method to determine optimal k
def plot_elbow(df_scaled: pd.DataFrame, k_range: range = range(2,11)) -> None:
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10)
                .fit(df_scaled).inertia_ for k in k_range]
    plt.plot(list(k_range), inertias, marker='o')
    optimal_k = 4
    optimal_inertia = inertias[k_range.index(optimal_k)]
    plt.scatter(optimal_k, optimal_inertia, color='red', s=100, zorder=5, label='Optimal k')
    plt.title('Elbow Method'); plt.xlabel('k'); plt.ylabel('Inertia'); plt.grid(True); plt.show()

# Plot cluster profiles for selected metrics
def plot_profiles(profile: pd.DataFrame, metrics: list) -> None:
    for m in metrics:
        plt.bar(profile['cluster'].astype(str), profile[m])
        plt.title(m); plt.xlabel('Cluster'); plt.grid(True); plt.show()

# Plot parallel coordinates for cluster centroids
def plot_parallel_centroids(df: pd.DataFrame, cluster_col: str, features: list) -> None:
    centroids = df.groupby(cluster_col)[features].mean().reset_index()
    centroids[cluster_col] = centroids[cluster_col].astype(str)
    plt.figure(figsize=(10,5))
    parallel_coordinates(centroids, cluster_col)
    plt.title('Centroid Profiles'); plt.xlabel('Features'); plt.ylabel('Value'); plt.grid(True); plt.show()


# Plot the most common RFM segment in each cluster
def plot_top_rfm_segments(df_raw: pd.DataFrame, cluster_col: str = 'cluster') -> None:
    """Plot the most common RFM segment in each cluster."""
    df = df_raw.copy()
    counts = (
        df.groupby([cluster_col, 'RFM_Segment'])
          .size()
          .reset_index(name='count')
    )
    
    top_segments = (
        counts.sort_values(['cluster', 'count'], ascending=[True, False])
              .drop_duplicates('cluster')
    )
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(top_segments[cluster_col].astype(str), top_segments['count'],
                   tick_label=top_segments[cluster_col])
    
    for bar, label in zip(bars, top_segments['RFM_Segment']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 2, label,
                 ha='center', va='bottom', fontsize=10)

    plt.title("Most Common RFM Segment per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.grid(False)
    plt.show()

# Plot a heatmap of RFM averages per cluster
def plot_rfm_heatmap(profile: pd.DataFrame) -> None:
    """Plot a heatmap of RFM averages per cluster."""
    rfm_cols = ['recency', 'frequency', 'monetary']
    heatmap_data = profile.set_index('cluster')[rfm_cols]

    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu')
    plt.title("RFM Averages per Cluster")
    plt.ylabel("Cluster")
    plt.xlabel("RFM Metric")
    plt.tight_layout()
    plt.show()

# Plot a heatmap of CLV-proxy metrics per cluster
def plot_clv_heatmap(profile: pd.DataFrame) -> None:
    """
    Plot a heatmap of CLV-proxy metrics per cluster.
    Expects `profile` to have columns:
      - 'cluster'
      - 'net_sales'
      - 'avg_order_value'
      - 'clv_proxy'
      - 'transactions'          (i.e. total_transactions mean)
      - 'recency_months'        (i.e. months_since_last mean)
    """
    clv_cols = ['net_sales', 'avg_order_value', 'clv_proxy', 'transactions', 'recency_months']
    heatmap_data = profile.set_index('cluster')[clv_cols]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={'label': 'Mean value'}
    )
    plt.title("CLV-Proxy Metrics per Cluster")
    plt.ylabel("Cluster")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.show()