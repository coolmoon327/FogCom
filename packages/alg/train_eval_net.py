import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

model_save_path = './data/classifier_model.pth'

class CustomClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class Trainer(object):
    def __init__(self, data, label, num_epochs=10, batch_size=64):
        self.data = data
        self.label = label
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self):
        # 设置输入和输出的维度
        input_size = len(self.data[0])
        output_size = 4  # 四个类别 {0, 1, 2, 3}

        # 创建模型
        model = CustomClassifier(input_size, output_size)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 将数据和标签转换为 PyTorch 的 Tensor 类型
        data_tensor = torch.Tensor(self.data)
        label_tensor = torch.Tensor(self.label).long()  # 注意，对于 CrossEntropyLoss，标签需要是 long 类型

        # 创建 DataLoader
        dataset = TensorDataset(data_tensor, label_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 训练模型
        for epoch in range(self.num_epochs):
            for data_batch, label_batch in dataloader:
                # 前向传播
                outputs = model(data_batch)

                # 计算损失
                loss = criterion(outputs, label_batch)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item()}')

            if epoch % 100 == 1:
                print(f"No. {epoch} epoch.")
                # 保存模型到指定路径
                torch.save(model.state_dict(), model_save_path)


