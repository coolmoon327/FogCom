import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

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
    
    def validate(self, dataloader, criterion):
        # 设置为评估模式
        self.eval()

        total_loss = 0.0
        correct_count = 0

        with torch.no_grad():
            for data, label in dataloader:
                # 前向传播
                outputs = self(data)

                # 计算损失
                loss = criterion(outputs, label)

                # 统计正确预测的样本数
                _, predicted = torch.max(outputs.data, 1)
                correct_count += (predicted == label).sum().item()

                total_loss += loss.item()

        # 恢复为训练模式
        self.train()

        # 计算平均损失和准确率
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = correct_count / len(dataloader.dataset)

        return avg_loss, accuracy

class Trainer(object):
    def __init__(self, num_epochs=10, batch_size=64, t_length=100):
        '''
        After initiating, you should set the dataset firstly.
        '''
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.t_length = t_length

        # 设置输入和输出的维度
        self.input_size = t_length + 5
        # targets(a list of ints) + p_link (a float) + p_s (a float) + bw (a float) + lt (a float) + csp (an int)
        self.output_size = 4  
        # 四个类别 {0, 1, 2, 3}

        # 创建模型
        self.model = CustomClassifier(self.input_size, self.output_size)

        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model_save_path = f'./data/classifier_model_{t_length}.pth'

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))

    def set_dataset(self, data, labels):
        # 将数据和标签转换为 PyTorch 的 Tensor 类型
        data_tensor = torch.Tensor(data)
        label_tensor = torch.Tensor(labels).long()  # 注意，对于 CrossEntropyLoss，标签需要是 long 类型
        
         # 分割训练集和验证集
        split_index = int(0.9 * len(data_tensor))
        train_dataset = TensorDataset(data_tensor[:split_index], label_tensor[:split_index])
        val_dataset = TensorDataset(data_tensor[split_index:], label_tensor[split_index:])

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def load_dataset(self, file_path = './data/database.pkl'):
        with open(file_path, 'rb') as file:
            database = pickle.load(file)

        # 提取 targets, state 和 strategy
        targets = [item['targets'][:self.t_length] for item in database]
        state = [item['state'] for item in database]
        labels = [item['strategy'] for item in database]

        # 合并 targets 和 state 为 data
        data = [torch.cat([torch.Tensor(target), torch.Tensor(s)], dim=0) for target, s in zip(targets, state)]

        # 将数据和标签转换为 PyTorch 的 Tensor 类型
        data_tensor = torch.stack(data)
        label_tensor = torch.Tensor(labels).long()

        self.set_dataset(data_tensor, label_tensor)

        # 打印 data 与 labels 的维度
        print("Data Dimensions:", data_tensor.shape)
        print("Label Dimensions:", label_tensor.shape)


    def train(self):
        model = self.model

        # 训练模型
        for epoch in range(self.num_epochs):
            for data_batch, label_batch in self.train_dataloader:
                # 前向传播
                outputs = model(data_batch)

                # 计算损失
                loss = self.criterion(outputs, label_batch)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item()}')

            if epoch % 10 == 1:
                print(f"No. {epoch} epoch.")
                # 保存模型到指定路径
                torch.save(model.state_dict(), self.model_save_path)
        
        torch.save(model.state_dict(), self.model_save_path)

    def validate(self):
        model = self.model
        
        val_loss, val_accuracy = model.validate(self.val_dataloader, self.criterion)
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
