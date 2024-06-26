# Get traning data for the follower strategy estimation network

import numpy as np
from ..env.fogcom.server import Server
from ..env.fogcom.user import User
from ..env.fogcom.utils import LinkCheck
import pickle
import os
import multiprocessing as mp
import numpy as np

class Get_Training_Data(object):
    def __init__(self, config: dict):
        self.config = config
        self.config['link_check'] = LinkCheck()
        self.pseudo_user = User(id=0, config=self.config)
        self.data_folder = "./data"  # 修改为你希望保存数据的文件夹路径

    def save_data(self, data, filename):
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        filepath = os.path.join(self.data_folder, filename)
        with open(filepath, 'wb') as file:
            pickle.dump(data, file)

    def load_data(self, filename):
        filepath = os.path.join(self.data_folder, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        else:
            return None

    def calculate_and_print_mean(self):
        if self.es_groups:
            print(f"{len(self.es_groups)} groups in total.")

            total_tasks = len(self.es_groups) * self.config['cand_num']
            sum_t = sum_s = sum_w = sum_sid = sum_b0 = sum_alpha = 0
            sum_c = sum_bw = sum_lt = sum_S = sum_p_c = sum_p_link = sum_p_s = 0
            sum_rd = sum_p_vm = 0

            for group in self.es_groups:
                task = group['task']
                sum_t += task.t
                sum_s += task.s
                sum_w += task.w
                sum_sid += task.sid
                sum_b0 += task.b0
                sum_alpha += task.alpha

                for server in group['servers']:
                    sum_c += server.c
                    sum_bw += server.bw
                    sum_lt += server.lt
                    sum_S += server.S
                    sum_p_c += server.p_c
                    sum_p_link += server.p_link
                    sum_p_s += server.p_s
                    sum_rd += server.rd
                    sum_p_vm += server.p_vm

            mean_t = sum_t / total_tasks
            mean_s = sum_s / total_tasks
            mean_w = sum_w / total_tasks
            mean_sid = sum_sid / total_tasks
            mean_b0 = sum_b0 / total_tasks
            mean_alpha = sum_alpha / total_tasks

            mean_c = sum_c / total_tasks
            mean_bw = sum_bw / total_tasks
            mean_lt = sum_lt / total_tasks
            mean_S = sum_S / total_tasks
            mean_p_c = sum_p_c / total_tasks
            mean_p_link = sum_p_link / total_tasks
            mean_p_s = sum_p_s / total_tasks
            mean_rd = sum_rd / total_tasks
            mean_p_vm = sum_p_vm / total_tasks

            print(f"Mean of t: {mean_t}")
            print(f"Mean of s: {mean_s}")
            print(f"Mean of w: {mean_w}")
            print(f"Mean of sid: {mean_sid}")
            print(f"Mean of b0: {mean_b0}")
            print(f"Mean of alpha: {mean_alpha}")

            print(f"Mean of c: {mean_c}")
            print(f"Mean of bw: {mean_bw}")
            print(f"Mean of lt: {mean_lt}")
            print(f"Mean of S: {mean_S}")
            print(f"Mean of p_c: {mean_p_c}")
            print(f"Mean of p_link: {mean_p_link}")
            print(f"Mean of p_s: {mean_p_s}")
            print(f"Mean of rd: {mean_rd}")
            print(f"Mean of p_vm: {mean_p_vm}")

    def generate_estimation_groups_and_save(self):
        print("Start generation.")

        config = self.config
        f_num = 1000

        print(f"Start generating {f_num} followers.")

        followers = [Server(id=i, config=config) for i in range(f_num)]

        print("Start generate raw groups.")

        # self.es_groups = []
        
        es_groups = []
        es_groups_std = []

        for _ in range(self.config['group_num']*500):
            task = self.pseudo_user.generate_task()
            servers = [Server(id=i, config=self.config) for i in range(self.config['cand_num'])]
            # self.es_groups.append({'task':task, 'servers':servers})
            
            target_ids = []
            for f in followers:
                target = f.select_storage(task, servers)
                if target is None:
                    continue
                target_ids.append(servers.index(target))
            
            es_groups_std.append(np.std(np.array(target_ids)))
            es_groups.append({'task':task, 'servers':servers})
        
        print("Finished generating raw groups.")

        print("Start sorting by std.")


        sorted_tuples = sorted(zip(es_groups, es_groups_std), key=lambda x: x[1], reverse=True)
        es_groups_sorted = [x[0] for x in sorted_tuples]
        self.es_groups = es_groups_sorted[:self.config['group_num']]

        print("Finished sorting.")
        
        std_sorted = [x[1] for x in sorted_tuples]
        # print(std_sorted[:self.config['group_num']])
        print(std_sorted)

        self.save_data(self.es_groups, "es_groups.pkl")

        print("Finished saving.")

        self.calculate_and_print_mean()
        

    def load_estimation_groups(self):
        es_groups = self.load_data("es_groups.pkl")
        if es_groups:
            self.es_groups = es_groups
            self.calculate_and_print_mean()
            return True
        else:
            return False

    def generate_training_data_and_save(self):
        # data (length = self.config['group_num'] + 5): 
        #       targets(a list of ints) + p_link (a float) + p_s (a float) + bw (a float) + lt (a float) + csp (an int)
        # label (length = 1): strategy - a number in {0,1,2,3}

        database = self.load_data("database.pkl")
        if database is None:
            database = []
        else:
            print("Loaded sccessfully")

        processing_num = 36

        pool = mp.Pool(processes=processing_num)
        results = []

        mp_num = 0

        while len(database) < self.config['training_data_num']:
            result = pool.apply_async(get_data, args=(self.config, self.es_groups))
            results.append(result)

            mp_num += 1
            if mp_num >= processing_num:
                pool.close()
                pool.join()

                for result in results:
                    data_list = result.get(timeout=1)
                    database += data_list

                print(f"No. {len(database)} data.")
                self.save_data(database, "database.pkl")
                # self.save_data(i, f"data_num_{i}")
                print("Finished saving.")
                
                pool = mp.Pool(processes=processing_num)
                mp_num = 0

def get_data(config, es_groups):
    data_list = []

    for i in range(10000):

        follower = Server(id=0, config=config)

        target_ids = []

        for tuple in es_groups:
            task = tuple['task']
            servers = tuple['servers']
            target = follower.select_storage(task, servers)
            if target is None:
                    continue
            target_ids.append(servers.index(target))
        
        state = [follower.p_link, follower.p_s, 
                    follower.bw, follower.lt, follower.csp]

        data = {'targets': target_ids, 'state': state, 'strategy':follower.strategy}

        data_list.append(data)

    return data_list
