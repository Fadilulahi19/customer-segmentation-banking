import matplotlib.pyplot as plt

def plot_elbow(inertia):
    plt.plot(range(2, 10), inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.savefig("visuals/elbow_plot.png")
    plt.close()

def plot_clusters(X_pca, labels):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("Customer Segments")
    plt.savefig("visuals/cluster_plot.png")
    plt.close()