import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data          
y = iris.target     
labels = iris.target_names 


print("Original shape:", X.shape)
print("Features:", iris.feature_names)
print()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled)

print("Reduced shape:", X_pca.shape)   
print()


explained = pca.explained_variance_ratio_

print("Variance explained by each PC:")
for i, var in enumerate(explained):
    print(f"  PC{i+1}: {var*100:.1f}%")

print(f"  Total kept: {sum(explained)*100:.1f}%")
print(f"  Information lost: {(1-sum(explained))*100:.1f}%")
print()


print("Feature contributions to each PC:")
for i, component in enumerate(pca.components_):
    print(f"\n  PC{i+1}:")
    for feature, loading in zip(iris.feature_names, component):
        bar = "█" * int(abs(loading) * 20)
        sign = "+" if loading > 0 else "-"
        print(f"    {sign}{bar} {feature}: {loading:.3f}")


colors = ['#7F77DD', '#1D9E75', '#D85A30']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Dimensionality Reduction with PCA — Iris Dataset", fontsize=14)

# Left: original data (first 2 features only)
ax1 = axes[0]
for i, label in enumerate(labels):
    mask = y == i
    ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=label,
                alpha=0.7, edgecolors='none', s=50)
ax1.set_title("Original data (2 of 4 features shown)")
ax1.set_xlabel(iris.feature_names[0])
ax1.set_ylabel(iris.feature_names[1])
ax1.legend()


ax2 = axes[1]
for i, label in enumerate(labels):
    mask = y == i
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=label,
                alpha=0.7, edgecolors='none', s=50)
ax2.set_title(f"After PCA: 4D → 2D\n({sum(explained)*100:.1f}% variance kept)")
ax2.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
ax2.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
ax2.legend()

plt.tight_layout()
plt.savefig("pca_result.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as pca_result.png")

pca_full = PCA()  
pca_full.fit(X_scaled)

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

print("\nCumulative variance by number of PCs:")
for i, cv in enumerate(cumulative_var):
    bar = "█" * int(cv * 30)
    print(f"  {i+1} PC(s): {bar} {cv*100:.1f}%")

n_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\nPCs needed to explain 95% variance: {n_95}")