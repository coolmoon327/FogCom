from packages.utils.utils import read_config
from packages.alg.get_training_data import Get_Training_Data
from packages.alg.train_eval_net import Trainer
import pandas as pd
import matplotlib.pyplot as plt

def generate_training_data():
    get_data = Get_Training_Data(config)
    # get_data.generate_estimation_groups_and_save()
    get_data.load_estimation_groups()
    get_data.generate_training_data_and_save()

def train(t_length=100):
    trainer = Trainer(num_epochs=3, batch_size=512, t_length=t_length)
    trainer.load_dataset()
    trainer.train()

def train_with_loading(t_length=100, val_in_batch=True):
    trainer = Trainer(num_epochs=2, batch_size=512, t_length=t_length)
    trainer.load_tensor_database()
    trainer.load_tensor_val()
    # trainer.load_model()
    return trainer.train(use_multi_datasets=False, val_in_batch=val_in_batch)

def eval(t_length=100):
    trainer = Trainer(num_epochs=1, batch_size=512, t_length=t_length)
    # trainer.load_dataset()
    trainer.load_tensor_val()
    trainer.load_model()
    return trainer.validate()

def dif_tl_train():
    val_accuracy_list = []

    # 迭代并将 val_accuracy 数据追加到 DataFrame 中的不同行
    tls = range(1, 21)
    for tl in tls:
        val_loss, val_accuracy = train_with_loading(tl, False)
        val_accuracy_list.append(val_accuracy)

    # 指定要保存的文件名
    file_name = './data/val_accuracy_data.csv'
    # 创建包含所有 val_accuracy 的 DataFrame
    data = pd.DataFrame({'t_length': list(tls), 'val_accuracy': val_accuracy_list})
    # 将 DataFrame 写入 Excel 文件
    data.to_csv(file_name, index=False)

def draw_dif_tl():
    # 指定要读取的文件名
    file_name = './data/val_accuracy_data.csv'

    # 从 CSV 文件中读取数据并存储在 DataFrame 中
    data = pd.read_csv(file_name)

    # 从 DataFrame 中提取 val_accuracy 列并转换为列表
    val_accuracy_list = data['val_accuracy'].tolist()

    tls = list(range(1, 21))

    plt.figure(figsize=(12, 8))

    plt.bar(tls, val_accuracy_list, label='Validate Dataset', zorder=1)
    
    # 添加图例和坐标轴标签
    # plt.legend(loc='lower right', fontsize=16)
    plt.xlabel("ST Length", fontsize=28, weight='bold')
    plt.ylabel("Accuracy", fontsize=28, weight='bold')

    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    plt.ylim(0.4, 1.05)
    plt.xlim(0, 21)
    plt.xticks(range(0,21))

    plt.grid(axis='y', which='both', linestyle='--', zorder=0.5)

    plt.savefig('./data/accuracy_tl_plot.png')

if __name__ == "__main__":
    config = read_config('config_F_estimation.yml')
    
    t_length=30

    # generate_training_data()
    # train(t_length)
    # train_with_loading(t_length)
    # eval(t_length)
    draw_dif_tl()

    
