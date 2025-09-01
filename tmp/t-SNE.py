import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

CLASS_NAMES = [
    "view1", "view2", "view3", "view4", "view5", "view6"
]

# ====================== 参数配置 ======================
selected_classes = [0, 1, 2, 3, 4, 5]  # 注意：这里使用的是CLASS_NAMES的索引
max_per_class = 800

# ====================== 数据加载 ======================
selected_embeddings = []
selected_class_indices = []

for cls in selected_classes:
    file_path = os.path.join("tmp", f"view_{cls+1}.npy")
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        embeddings = np.load(file_path, allow_pickle=True)
        embeddings = np.array(embeddings)
        print(embeddings.shape)
        if embeddings.shape[0] > 0:
            actual_take = min(max_per_class, embeddings.shape[0])
            selected_embeddings.append(embeddings[:actual_take])
            selected_class_indices.extend([cls] * actual_take)

if not selected_embeddings:
    raise ValueError("数据为空")

selected_embeddings = np.concatenate(selected_embeddings, axis=0)
selected_class_indices = np.array(selected_class_indices)

# ====================== t-SNE降维 ======================
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(selected_embeddings)

# ====================== 可视化修改版 ======================
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', len(selected_classes))

# 核心修改点 ▼▼▼ (仅修改标签显示方式)
# 1. 图例标签映射
legend_handles = [
    plt.Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor=colors(i),
               markersize=10,
               label=CLASS_NAMES[cls])  # 核心修改：使用预设名称
    for i, cls in enumerate(selected_classes)
]

# 2. 散点标签绑定
valid_classes = np.unique(selected_class_indices)
for i, cls in enumerate(valid_classes):
    mask = (selected_class_indices == cls)
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                color=colors(i),
                alpha=0.6,
                s=50,
                label=CLASS_NAMES[cls])  # 同步修改散点标签
# 核心修改点 ▲▲▲

# 保留原有设置 ▼
plt.xticks([])
plt.yticks([])
plt.gca().tick_params(axis='both', which='both', length=0)

legend = plt.legend(
    handles=legend_handles,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    borderaxespad=0.3
)

plt.tight_layout()
plt.show()
