"""
t-SNE map of representative tiles across the six GMM prototypes (Pheno-MYCN).

Loads the per-prototype representative-tile tables (``top_df_proto_0..5.csv``),
embeds their features with t-SNE and renders the combined scatter coloured by
prototype / cluster label used in the phenotype-space figure.

This is a stand-alone figure script (no argparse): edit the ``csv_paths`` list
and the ``plt.savefig`` targets — both placeholders below — to point at your own
representative-tile tables and output location.

Part of Pheno-MYCN: interpretable histological phenotype discovery associated
with MYCN amplification in paediatric neuroblastoma.

Author:  Dr Olga Fourkioti  (https://github.com/olgarithmics)

License: GPL-3.0 (see the LICENSE file at the repository root).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE


csv_paths = [
    '/path/to/results/fold_9/images/top_df_proto_0.csv',
    '/path/to/results/fold_9/images/top_df_proto_1.csv',
    '/path/to/results/fold_9/images/top_df_proto_2.csv',
    '/path/to/results/fold_9/images/top_df_proto_3.csv',
    '/path/to/results/fold_9/images/top_df_proto_4.csv',
    '/path/to/results/fold_9/images/top_df_proto_5.csv',

    # '/path/to/results/fold_9/images/prototype_2/features/cell_info_updated.csv',
    # '/path/to/results/fold_9/images/prototype_3/features/cell_info_updated.csv',
    # '/path/to/results/fold_9/images/prototype_4/features/cell_info_updated.csv',
    # '/path/to/results/fold_9/images/prototype_5/features/cell_info_updated.csv'
]

# Load and label each CSV
prototype_dfs = []
for i, path in enumerate(csv_paths, start=1):
    df = pd.read_csv(path)
    df['prototype'] = i  # Label prototype as 1-6
    prototype_dfs.append(df)

# Combine all prototypes
all_patches_df = pd.concat(prototype_dfs, ignore_index=True)
print("✅ Combined DataFrame shape:", all_patches_df.shape)


# # Keep numeric columns only (excluding prototype)
feature_columns = all_patches_df.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in feature_columns if col != 'prototype']
features_df = all_patches_df[feature_columns]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(features_df)

# Check for any remaining NaNs
print("✅ NaNs in final data:", np.isnan(X_imputed).sum())

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_imputed)

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# # Create a new figure
# fig, ax = plt.subplots(figsize=(12, 10))
#
# for idx, (x0, y0, path) in enumerate(zip(X_tsne[:, 0], X_tsne[:, 1], all_patches_df['image_path'])):
#     try:
#         img = Image.open(path)
#         img = img.resize((10, 10))  # adjust thumbnail size
#         imagebox = OffsetImage(img, zoom=1)
#         ab = AnnotationBbox(imagebox, (x0, y0), frameon=False)
#         ax.add_artist(ab)
#     except Exception as e:
#         print(f"⚠️ Could not open {path}: {e}")
#
# ax.set_xlim(X_tsne[:, 0].min()-5, X_tsne[:, 0].max()+5)
# ax.set_ylim(X_tsne[:, 1].min()-5, X_tsne[:, 1].max()+5)
# ax.set_xlabel('t-SNE1')
# ax.set_ylabel('t-SNE2')
# ax.set_title('t-SNE of Cell-level Features (Patches as Images)')
#
# plt.tight_layout()
# #plt.savefig('/path/to/results/fold_9/images/tsne_proto_patches.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Plot

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
import numpy as np
from scipy.spatial.distance import cdist

# Clean style
mpl.rcParams.update({
    'font.size': 12,
    'figure.dpi': 200
})

plt.figure(figsize=(7, 5.5))
cmap = get_cmap('Set2', 6)

for p in range(1, 7):
    mask = all_patches_df['prototype'] == p
    cluster_points = X_tsne[mask]

    # Plot the cluster
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        color=cmap(p - 1),
        s=25,
        alpha=0.7,
        edgecolor='white',
        linewidth=0.3
    )

    # Find the point closest to the centroid
    centroid = cluster_points.mean(axis=0)
    dists = cdist([centroid], cluster_points)
    closest_idx = np.argmin(dists)
    label_point = cluster_points[closest_idx]

    # Place label directly on point (no box)
    plt.text(
        label_point[0],
        label_point[1],
        str(p),
        fontsize=10,
        fontweight='bold',
        color='black',
        ha='center',
        va='center'
    )

# Clean look
plt.xticks([])
plt.yticks([])
plt.xlabel('')
plt.ylabel('')
plt.box(False)
plt.tight_layout()

plt.savefig('/path/to/results/fold_9/images/tsne_cluster_labels_nobox.png', dpi=300, bbox_inches='tight')
plt.show()


# Save for publication
#plt.savefig('/path/to/results/fold_9/images/tsne_proto.png', dpi=300, bbox_inches='tight')






#
# plt.figure(figsize=(8, 6))
# cmap = plt.cm.get_cmap('tab10', 6)
#
# for p in range(1, 7):  # prototypes 1-6
#     mask = all_patches_df['prototype'] == p
#     plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=cmap(p-1), label=f'Prototype {p}', s=20)
#
# plt.xlabel('t-SNE1')
# plt.ylabel('t-SNE2')
# plt.title('t-SNE of Cell-level Features')
# plt.legend(title='Prototype', loc='best')
#
# #plt.savefig('/path/to/results/fold_9/images/tsne_proto.png', dpi=300, bbox_inches='tight')
# plt.tight_layout()
# plt.show()

# from sklearn.cluster import KMeans
#
# # Parameters
# examples_per_subcluster = 10  # images to sample from each subcluster
# n_subclusters = 3  # how many subclusters per prototype
# thumbnail_size = (32, 32)
#
# fig, ax = plt.subplots(figsize=(12, 10))
#
# for prototype in range(1, 7):  # loop over prototype labels
#     # Filter data for this prototype
#     mask = all_patches_df['prototype'] == prototype
#     tsne_subset = X_tsne[mask]
#     image_paths = all_patches_df.loc[mask, 'image_path'].values
#
#     if len(tsne_subset) < n_subclusters:
#         continue  # not enough data to cluster
#
#     # Cluster in t-SNE space
#     kmeans = KMeans(n_clusters=n_subclusters, random_state=0).fit(tsne_subset)
#     labels = kmeans.labels_
#
#     for cluster_id in range(n_subclusters):
#         cluster_mask = labels == cluster_id
#         cluster_indices = np.where(mask)[0][cluster_mask]
#
#         # Randomly select a few from this subcluster
#         selected_indices = np.random.choice(cluster_indices, size=min(examples_per_subcluster, len(cluster_indices)),
#                                             replace=False)
#
#         for idx in selected_indices:
#             x0, y0 = X_tsne[idx]
#             path = all_patches_df.iloc[idx]['image_path']
#             try:
#                 img = Image.open(path).resize(thumbnail_size)
#                 imagebox = OffsetImage(img, zoom=1)
#                 ab = AnnotationBbox(imagebox, (x0, y0), frameon=False, pad=0.2)
#                 ax.add_artist(ab)
#                 # ax.text(x0, y0, str(prototype), color='white', fontsize=8,
#                 #         ha='center', va='center', weight='bold',
#                 #         backgroundcolor='black', alpha=0.6)
#             except Exception as e:
#                 print(f"⚠️ Could not open {path}: {e}")
#
#
# ax.set_xlim(X_tsne[:, 0].min() - 5, X_tsne[:, 0].max() + 5)
# ax.set_ylim(X_tsne[:, 1].min() - 5, X_tsne[:, 1].max() + 5)
# ax.set_xlabel('t-SNE1')
# ax.set_ylabel('t-SNE2')
# ax.set_title('t-SNE with Prototype Subclusters (Representative Patch Images)')
# #plt.grid(True)
# plt.tight_layout()
# plt.savefig('/path/to/results/fold_9/images/tsne_proto_subclusters.png', dpi=300)
# plt.show()

