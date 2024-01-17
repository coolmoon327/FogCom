import pickle

split_num = 30

file_path = './backup/database.pkl'

with open(file_path, 'rb') as file:
    database = pickle.load(file)

all_num = len(database)

for i in range(split_num):
    start = int(all_num * i / split_num )
    end = int(all_num * (i+1) / split_num)
    try:
        new_db = database[start:end]
        print(f"No.{i} has {len(new_db)} to save.")
        new_path = f'./splited/database{i}.pkl'
        with open(new_path, 'wb') as file:
            pickle.dump(new_db, file)
    except:
        print(f"Cannot save No.{start} to No.{end} of {all_num}.")