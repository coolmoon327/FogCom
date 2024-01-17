import pickle
import torch

def training_data():
    split_num = 30

    for i in range(split_num):
        file_path = f'./splited/database{i}.pkl'

        with open(file_path, 'rb') as file:
            print(f"Start loading dataset{i}.")
            database = pickle.load(file)

            # 提取 targets, state 和 strategy
            data = [torch.cat([torch.Tensor(item['targets']).float(), torch.Tensor(item['state']).float()], dim=0) for item in database]
            labels = [item['strategy'] for item in database]
            del database
            print("Finished extracting data.")

            # 将数据和标签转换为 PyTorch 的 Tensor 类型
            data_tensor = torch.stack(data).to(torch.float32)
            del data
            label_tensor = torch.Tensor(labels).long()
            del labels

            torch.save(data_tensor, f'./transformed/data{i}.pt')
            torch.save(label_tensor, f'./transformed/labels{i}.pt')
            del data_tensor, label_tensor

def val_data():
    file_path = f'./splited/database29.pkl'
    with open(file_path, 'rb') as file:
        database = pickle.load(file)
        split_index = int(0.01 * len(database))
        database = database[:split_index]

        data = [torch.cat([torch.Tensor(item['targets']).float(), torch.Tensor(item['state']).float()], dim=0) for item in database]
        labels = [item['strategy'] for item in database]

        data_tensor = torch.stack(data).to(torch.float32)
        label_tensor = torch.Tensor(labels).long()

        torch.save(data_tensor, f'./transformed/val/data.pt')
        torch.save(label_tensor, f'./transformed/val/labels.pt')
        

val_data()
    