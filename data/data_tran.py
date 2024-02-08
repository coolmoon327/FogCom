import pickle
import torch

def training_data(split_num = 30):

    for i in range(max(1, split_num)):
        if split_num == 0:
            file_path = f'./database.pkl'
        else:
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

def val_data(split_num = 30):
    if split_num == 0:
        file_path = f'./database.pkl'
    else:
        file_path = f'./splited/database{split_num-1}.pkl'
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

def all_data():
    file_path = f'./database.pkl'
    with open(file_path, 'rb') as file:
        print(f"Start loading dataset.")
        database = pickle.load(file)

        # 提取 targets, state 和 strategy
        data = [torch.cat([torch.Tensor(item['targets']).float(), torch.Tensor(item['state']).float()], dim=0) for item in database]
        labels = [item['strategy'] for item in database]
        del database
        print("Finished extracting data.")

        # 将数据和标签转换为 PyTorch 的 Tensor 类型
        data_tensor = torch.stack(data).to(torch.float32)
        label_tensor = torch.Tensor(labels).long()

        # 分割训练与测试
        split_index = int(0.9 * len(data_tensor))

        torch.save(data_tensor[:split_index], f'./transformed/data.pt')
        torch.save(label_tensor[:split_index], f'./transformed/labels.pt')

        torch.save(data_tensor[split_index:], f'./transformed/val/data.pt')
        torch.save(label_tensor[split_index:], f'./transformed/val/labels.pt')

if __name__ == "__main__":
    # val_data()
    all_data()
    