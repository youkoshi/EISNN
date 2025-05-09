import numpy as np
import matplotlib.pyplot as plt

# 假设你的均值和方差数据
mean_values = np.random.rand(3, 5000)  # 用随机数据代替，实际为你的均值数据
std_values = np.random.rand(3, 5000)   # 用随机数据代替，实际为你的方差数据

# 为了绘制其中一个变量的结果，比如第一个变量（索引为 0）
x = np.arange(5000)

# 创建绘图
plt.figure(figsize=(10, 6))

# 绘制均值线
plt.plot(x, mean_values[0, :], label='Mean', color='b', lw=2)

# 绘制均值 ± 3 标准差的范围（填充）
plt.fill_between(x, mean_values[0, :] - 3 * std_values[0, :], mean_values[0, :] + 3 * std_values[0, :],
                 color='b', alpha=0.2, label=r'$\pm 3 \sigma$')

# 添加标题和标签
plt.title('Mean and ± 3 Standard Deviation for Variable 1', fontsize=14)
plt.xlabel('Data Points', fontsize=12)
plt.ylabel('Value', fontsize=12)

# 添加图例
plt.legend()

# 设置网格
plt.grid(True)

# 调整边距
plt.tight_layout()

# 显示图形
plt.show()

