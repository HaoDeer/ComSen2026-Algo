import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 数据准备 =================
print("正在从网络加载 Air Passengers 数据集...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=[1])
data = df.values.astype('float32')

# 归一化处理 (Min-Max)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# 构造滑动窗口数据 (用过去 12 个月预测下 1 个月)
def create_dataset(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 12
X, Y = create_dataset(data_normalized, look_back)

# 划分训练集和测试集 (80% 训练, 20% 测试)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 转为 PyTorch Tensor (LSTM 需要的形状: [样本数, 序列长度, 特征数])
X_train = torch.FloatTensor(X_train).unsqueeze(-1)
Y_train = torch.FloatTensor(Y_train).unsqueeze(-1)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
Y_test = torch.FloatTensor(Y_test).unsqueeze(-1)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)

# ================= 2. 搭建 LSTM 模型 =================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # 取最后一个时间步的输出
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ================= 3. 训练模型 =================
epochs = 100
loss_list = []
print("开始训练 LSTM 模型...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    loss_list.append(avg_loss)
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# ================= 4. 预测与评估 =================
model.eval()
with torch.no_grad():
    train_predict = model(X_train).numpy()
    test_predict = model(X_test).numpy()

# 反归一化还原数据
train_predict = scaler.inverse_transform(train_predict)
Y_train_orig = scaler.inverse_transform(Y_train.numpy())
test_predict = scaler.inverse_transform(test_predict)
Y_test_orig = scaler.inverse_transform(Y_test.numpy())

# 计算评价指标 RMSE
rmse = np.sqrt(np.mean((test_predict - Y_test_orig)**2))
print(f"\n测试集 RMSE (均方根误差): {rmse:.2f}")

# ================= 5. 可视化 =================
plt.figure(figsize=(12, 5))

# (1) Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(loss_list, color='red')
plt.title('Model Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# (2) 预测曲线对比图
plt.subplot(1, 2, 2)
# 构造完整的时间轴
all_predict = np.concatenate((np.full(look_back, np.nan), train_predict.flatten(), test_predict.flatten()))
plt.plot(data, label='真实值 (True)', color='blue')
plt.plot(all_predict, label='预测值 (Predict)', color='orange', linestyle='--')
plt.title('Air Passengers 时序预测对比')
plt.xlabel('时间 (月)')
plt.ylabel('乘客数量')
plt.legend()

plt.tight_layout()
plt.savefig('task4_lstm_results.png')
print("SUCCESS: 结果图已保存为 task4_lstm_results.png")
plt.show()