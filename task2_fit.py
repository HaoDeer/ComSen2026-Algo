import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题（如果你电脑里有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据 (真实公式 + 噪声)
N = 1000
x1 = torch.rand(N) * 6 - 3  # 在 -3 到 3 之间随机
x2 = torch.rand(N) * 6 - 3
noise = torch.rand(N) * 6 - 3

# 真实 y 值公式
y_true = 2*(x1**2) + 1.5*x1*x2 + 3*torch.sin(x1) + 0.5*torch.cos(2*x2) + torch.exp(-0.5*x1) + 0.8*x2 + 2026
y_data = y_true + noise

# 2. 定义我们要学习的参数 (初始值全设为 0)
a1 = torch.tensor(0.0, requires_grad=True)
a2 = torch.tensor(0.0, requires_grad=True)
a3 = torch.tensor(0.0, requires_grad=True)
a4 = torch.tensor(0.0, requires_grad=True)
a5 = torch.tensor(0.0, requires_grad=True)
a6 = torch.tensor(0.0, requires_grad=True)
b  = torch.tensor(0.0, requires_grad=True)

# 3. 定义优化器 (使用 Adam，比 SGD 收敛更快)
optimizer = torch.optim.Adam([a1, a2, a3, a4, a5, a6, b], lr=0.1)

# 4. 开始梯度下降训练
epochs = 3000
for epoch in range(epochs):
    # 前向传播：用我们现在的参数计算预测值
    y_pred = a1*(x1**2) + a2*x1*x2 + a3*torch.sin(x1) + a4*torch.cos(2*x2) + torch.exp(a5*x1) + a6*x2 + b
    
    # 计算损失 (均方误差 MSE)
    loss = torch.mean((y_pred - y_data)**2)
    
    # 反向传播与更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. 输出拟合结果
print("\n--- 拟合结果 ---")
print(f"真实: a1=2.0, a2=1.5, a3=3.0, a4=0.5, a5=-0.5, a6=0.8, b=2026")
print(f"预测: a1={a1.item():.3f}, a2={a2.item():.3f}, a3={a3.item():.3f}, a4={a4.item():.3f}, a5={a5.item():.3f}, a6={a6.item():.3f}, b={b.item():.3f}")

# 计算 R^2 (决定系数)
ss_res = torch.sum((y_data - y_pred)**2)
ss_tot = torch.sum((y_data - torch.mean(y_data))**2)
r2 = 1 - ss_res / ss_tot
print(f"决定系数 R^2: {r2.item():.4f}")

# 6. 绘图 (3D 图)
fig = plt.figure(figsize=(12, 6))

# 创建网格数据用于画光滑的 3D 曲面
x1_grid, x2_grid = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
x1_g = torch.tensor(x1_grid.flatten(), dtype=torch.float32)
x2_g = torch.tensor(x2_grid.flatten(), dtype=torch.float32)

# 真实曲面 (无噪声)
y_real_grid = (2*(x1_g**2) + 1.5*x1_g*x2_g + 3*torch.sin(x1_g) + 0.5*torch.cos(2*x2_g) + torch.exp(-0.5*x1_g) + 0.8*x2_g + 2026).reshape(x1_grid.shape).detach().numpy()

# 拟合曲面
y_fit_grid = (a1*(x1_g**2) + a2*x1_g*x2_g + a3*torch.sin(x1_g) + a4*torch.cos(2*x2_g) + torch.exp(a5*x1_g) + a6*x2_g + b).reshape(x1_grid.shape).detach().numpy()

# 画真实图
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x1_grid, x2_grid, y_real_grid, cmap='viridis', alpha=0.8)
ax1.set_title(f'Ground Truth Function\n\(R^2\) = {r2.item():.4f}')
ax1.set_xlabel('x1'); ax1.set_ylabel('x2'); ax1.set_zlabel('y')

# 画拟合图
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x1_grid, x2_grid, y_fit_grid, cmap='plasma', alpha=0.8)
ax2.set_title(f'Fitted Function\n\(R^2\) = {r2.item():.4f}')
ax2.set_xlabel('x1'); ax2.set_ylabel('x2'); ax2.set_zlabel('y')

plt.tight_layout()
plt.savefig('task2_fit_3d.png', dpi=150)
print("SUCCESS: 图像已保存为 task2_fit_3d.png")
plt.show()