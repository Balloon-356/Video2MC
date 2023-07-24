import numpy as np
import matplotlib.pyplot as plt

# 导入数据
import pickle
with open("output_3Dpose_npy/kun_1280x720_30fps_0-14_0-32.npy", 'rb') as file:
    data = np.load(file)
with open("skeleton.npy", 'rb') as file:
    skeleton = pickle.load(file)

# 提取第0帧坐标
xyz_0 = data[0]

# 创建3D坐标系
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# 设置俯仰角和方位角
ax.view_init(elev=0., azim=70)

# 绘制3D点
radius = 1.7
ax.scatter(xyz_0[:, 0], xyz_0[:, 1], xyz_0[:, 2])
# 添加文本标记
for i in range(xyz_0.shape[0]):
    ax.text(xyz_0[i, 0], xyz_0[i, 1], xyz_0[i, 2], str(i), fontsize=10)

# 绘制两点间的线段
for num1 in range(xyz_0.shape[0]):
    parent = skeleton._parents
    num2 = parent[num1]
    if num2 != -1:
        x1, y1, z1 = xyz_0[num1, :]
        x2, y2, z2 = xyz_0[num2, :]
        ax.plot([x1, x2], [y1, y2], [z1, z2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_xlim3d([-radius/2, radius/2])
# ax.set_ylim3d([-radius/2, radius/2])
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])

# 保存图像
plt.savefig('plot.png')
