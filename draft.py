import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances

# 假设 a 是 16x101 的数据，每一行是一个样本（按顺序排列）
# 这里用随机数据模拟，并人为制造一个异常
np.random.seed(0)
a = np.cumsum(np.random.randn(16, 101), axis=1)  # 连续变化的样本
# 人为制造异常：例如第 8 个样本中某些峰消失或突变
a[7, 40:60] -= 20

# 使用 KernelPCA 将数据降至 2 维（也可以选3维）
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.01)  
a_kpca = kpca.fit_transform(a)

# 计算每个样本在低维空间中的邻近距离
# 这里计算所有样本之间的欧式距离
dist_matrix = pairwise_distances(a_kpca)

# 选择 k 近邻，这里取 k=3（不包括自身）
k = 3
anomaly_scores = []
for i in range(a_kpca.shape[0]):
    # 距离矩阵第 i 行，排序后排除第一个（0距离，即自身）
    sorted_dists = np.sort(dist_matrix[i])[1:k+1]
    score = np.mean(sorted_dists)
    anomaly_scores.append(score)
anomaly_scores = np.array(anomaly_scores)

# 设定阈值：这里简单采用均值加上1个标准差作为阈值
threshold = anomaly_scores.mean() + anomaly_scores.std()

# 找出异常样本
anomaly_idx = np.where(anomaly_scores > threshold)[0]
print("异常样本索引：", anomaly_idx)

# 绘制低维表示，标出异常样本
plt.figure(figsize=(8,6))
plt.scatter(a_kpca[:,0], a_kpca[:,1], c='blue', label='Normal samples')
plt.scatter(a_kpca[anomaly_idx,0], a_kpca[anomaly_idx,1], 
            c='red', label='Anomaly', s=100, marker='x')
for i, txt in enumerate(range(a.shape[0])):
    plt.annotate(txt, (a_kpca[i,0], a_kpca[i,1]))
plt.xlabel("KernelPCA Component 1")
plt.ylabel("KernelPCA Component 2")
plt.title("KernelPCA 降维结果及异常检测")
plt.legend()
plt.show()
