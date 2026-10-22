import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据 (MNIST 手写数字)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# 如果下载慢，请耐心等待一下
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 搭建简单的多层感知机 (MLP)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 输入 28*28=784, 两个隐藏层, 输出 10 个类别(0-9)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28) # 展平图片
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练模型
epochs = 5 # CPU 跑 5 轮很快，大概几分钟
loss_list = []
train_acc_list = []
test_acc_list = []

print("开始训练... 请稍候（下载数据集可能需要一点时间）")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    loss_list.append(running_loss / len(train_loader))
    train_acc_list.append(train_acc)
    
    # 测试准确率
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    test_acc_list.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_list[-1]:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

# 4. 绘制 Loss 和 准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), loss_list, label='Train Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_acc_list, label='Train Acc', color='blue')
plt.plot(range(1, epochs+1), test_acc_list, label='Test Acc', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('task2_mlp_curves.png')
print("SUCCESS: 曲线图已保存为 task2_mlp_curves.png")

# 5. 可视化至少 3 张测试图片结果
dataiter = iter(test_loader)
images, labels = next(dataiter)
images_vis = images[:3]
labels_vis = labels[:3]

model.eval()
with torch.no_grad():
    outputs = model(images_vis)
    _, predicted = torch.max(outputs, 1)

plt.figure(figsize=(8, 3))
for i in range(3):
    plt.subplot(1, 3, i+1)
    # 还原图片以便显示
    img = images_vis[i].numpy().squeeze()
    img = img * 0.5 + 0.5 # 反归一化
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {labels_vis[i].item()}\nPred: {predicted[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('task2_mlp_vis.png')
print("SUCCESS: 预测可视化已保存为 task2_mlp_vis.png")
plt.show()