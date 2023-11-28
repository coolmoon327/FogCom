from packages.utils.utils import read_config
from packages.alg.get_training_data import Get_Training_Data

if __name__ == "__main__":
    config = read_config('config_F_estimation.yml')

    get_data = Get_Training_Data(config)

    # get_data.generate_estimation_groups_and_save()\
    get_data.load_estimation_groups()
    get_data.generate_training_data_and_save()