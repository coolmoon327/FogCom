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
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 64)
        self.fc6 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.softmax(x)
        return x
    
    def validate(self, dataloader, criterion, device="cpu"):
        # 设置为评估模式
        self.eval()

        total_loss = 0.0
        correct_count = 0

        with torch.no_grad():
            for data, label in dataloader:
                # 将数据移至GPU
                data = data.to(device)
                label = label.to(device)

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

        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        # 创建模型
        self.model = CustomClassifier(self.input_size, self.output_size)

        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.000001)

        self.model_save_path = f'./data/classifier_model_{t_length}.pth'

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))

    def load_dataset(self, file_path = './data/database.pkl'):
        print("Start loading dataset.")

        with open(file_path, 'rb') as file:
            database = pickle.load(file)
        
        print("Finished loading data.")

        # 提取 targets, state 和 strategy
        data = [torch.cat([torch.Tensor(item['targets'][:self.t_length]).float(), torch.Tensor(item['state']).float()], dim=0) for item in database]
        labels = [item['strategy'] for item in database]
        del database
        print("Finished extracting data.")

        # 将数据和标签转换为 PyTorch 的 Tensor 类型
        data_tensor = torch.stack(data).to(torch.float32)
        del data

        label_tensor = torch.Tensor(labels).long()
        del labels

        # 打印 data 与 labels 的维度
        print("Data Dimensions:", data_tensor.shape)
        print("Label Dimensions:", label_tensor.shape)

        print("Start making dataloader.")
        
         # 分割训练集和验证集
        split_index = int(0.99 * len(data_tensor))
        train_dataset = TensorDataset(data_tensor[:split_index], label_tensor[:split_index])
        val_dataset = TensorDataset(data_tensor[split_index:], label_tensor[split_index:])

        del data_tensor,label_tensor
        print("Finished splitting data.")

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)


    def load_tensor_database(self, dir_path='./data/transformed', file_num=0):
        data_path = f'{dir_path}/data{file_num}.pt'
        labels_path = f'{dir_path}/labels{file_num}.pt'
        data = torch.load(data_path)
        labels = torch.load(labels_path)
        if self.t_length < 100:
            data = torch.cat([data[:, :self.t_length], data[:, 100:]], dim=1)

        train_dataset = TensorDataset(data, labels)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def load_tensor_val(self, dir_path='./data/transformed/val'):
        data_path = f'{dir_path}/data.pt'
        labels_path = f'{dir_path}/labels.pt'
        data = torch.load(data_path)
        labels = torch.load(labels_path)
        if self.t_length < 100:
            data = torch.cat([data[:, :self.t_length], data[:, 100:]], dim=1)

        val_dataset = TensorDataset(data, labels)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, use_multi_datasets=False):
        model = self.model.to(self.device)
        optimizer = self.optimizer

        splited_num = 20
        single_db_training_num = int(self.num_epochs/splited_num)
        db_num = 0

        print("Start training.")

        # 训练模型
        for epoch in range(self.num_epochs):
            if use_multi_datasets:
                if epoch % single_db_training_num == 0:
                    if db_num:
                        del self.train_dataloader
                    self.load_tensor_database(file_num=db_num)
                    db_num = (db_num + 1) % splited_num

            for data_batch, label_batch in self.train_dataloader:
                # 将数据移至GPU
                data_batch = data_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                # 前向传播
                outputs = model(data_batch)

                # 计算损失
                loss = self.criterion(outputs, label_batch)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item()}')
            self.validate()

            if epoch % 10 == 1:
                # print(f"No. {epoch} epoch.")
                # self.validate()
                # 保存模型到指定路径
                torch.save(model.state_dict(), self.model_save_path)
        
        torch.save(model.state_dict(), self.model_save_path)

    def validate(self):
        model = self.model
        
        val_loss, val_accuracy = model.validate(self.val_dataloader, self.criterion, self.device)
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
