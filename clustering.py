import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def apply_clustering(features):
    """
    Apply K-Means clustering to the data.
    """
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Create a DataFrame for the reduced features
    reduced_rfm = pd.DataFrame(features, columns=[f'PCA{i+1}' for i in range(features.shape[1])])
    reduced_rfm['Cluster'] = cluster_labels

    print("Clustering applied.")
    return reduced_rfm

def visualize_clusters(rfm_with_clusters):
    """
    Visualize the clusters using scatter plots for reduced features (PCA1, PCA2).
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=rfm_with_clusters['PCA1'],
        y=rfm_with_clusters['PCA2'],
        hue=rfm_with_clusters['Cluster'],
        palette='Set2',
        s=100
    )
    plt.title("Clusters: PCA1 vs PCA2", fontsize=16)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(title="Cluster")
    plt.show()

def summarize_clusters(rfm_with_clusters):
    """
    Summarize the clusters by calculating mean values for each PCA component.
    """
    summary = rfm_with_clusters.groupby('Cluster').mean()
    print("Cluster Summary:")
    print(summary)
    return summary

def visualize_cluster_summary(cluster_summary):
    """
    Create bar charts for average PCA values for each cluster.
    """
    cluster_summary.plot(kind="bar", figsize=(10, 6))
    plt.title("Cluster-wise Average PCA Values", fontsize=16)
    plt.xlabel("Cluster", fontsize=14)
    plt.ylabel("Average Values (Scaled)", fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title="PCA Metrics", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def visualize_cluster_sizes(rfm_with_clusters):
    """
    Create a pie chart showing the size of each cluster.
    """
    cluster_counts = rfm_with_clusters['Cluster'].value_counts()
    cluster_counts.plot(kind="pie", autopct='%1.1f%%', figsize=(8, 8), startangle=90, colors=sns.color_palette("Set2"))
    plt.title("Cluster Size Distribution", fontsize=16)
    plt.ylabel("")  # Remove the default ylabel
    plt.show()

def apply_pca(features, n_components=2):
    """
    Apply PCA to reduce dimensionality of the data.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    return reduced_features

def find_optimal_k(features):
    """
    Use the Elbow Method to find the optimal number of clusters.
    """
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method: Optimal k', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Inertia', fontsize=14)
    plt.show()

def evaluate_clustering(features, cluster_labels):
    """
    Evaluate clustering performance using silhouette score.
    """
    score = silhouette_score(features, cluster_labels)
    print(f"Silhouette Score: {score}")
    return score
