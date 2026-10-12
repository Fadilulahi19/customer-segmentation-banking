from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def find_optimal_k(X):
    inertia = []
    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)
    return inertia

def fit_kmeans(X, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def apply_pca(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(X)