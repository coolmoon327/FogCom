from packages.utils.utils import read_config
from packages.alg.get_training_data import Get_Training_Data
from packages.alg.train_eval_net import Trainer

def generate_training_data():
    get_data = Get_Training_Data(config)
    # get_data.generate_estimation_groups_and_save()
    get_data.load_estimation_groups()
    get_data.generate_training_data_and_save()

def train(t_length=100):
    trainer = Trainer(num_epochs=1000, batch_size=1024, t_length=t_length)
    trainer.load_dataset()
    trainer.train()

def eval(t_length=100):
    trainer = Trainer(num_epochs=1000, batch_size=1024, t_length=t_length)
    trainer.load_dataset()
    trainer.load_model()
    trainer.validate()

if __name__ == "__main__":
    config = read_config('config_F_estimation.yml')

    generate_training_data()
    # train()