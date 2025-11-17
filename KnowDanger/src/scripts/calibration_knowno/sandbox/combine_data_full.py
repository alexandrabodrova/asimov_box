import pickle
import os
import random

# combine final answer
data_dir_1 = 'data/saycan_full/archive/saycan_v1/collect/answer_no_probe/answer.pkl'  # easy
data_dir_2 = 'data/saycan_final/archive/saycan_v3/collect/answer/single_data'  # harder
output_path = 'data/saycan_full/collect/answer/answer.pkl'

num_data = 800

with open(data_dir_1, 'rb') as f:
    data_all = pickle.load(f)

# data_all = []
# for data_ind in range(700):
#     with open(os.path.join(data_dir_1, f'{data_ind}.pkl'), 'rb') as f:
#         data = pickle.load(f)

#     # true_label = data['true_label']
#     # if len(
#     #     true_label[0]
#     # ) == 0:  # check if accidentally put multiple latters for a label
#     #     print(data['request'])
#     #     print(data['mc_prompt'])
#     #     print(data['true_intent'])

#     # print(data['task_category'])
#     # if data['task_category'] == 'unambiguous_task':
#     #     if random.random() > 0.5:
#     #         continue
#     # if len(true_label) > 1:
#     #     continue
#     # if len(true_label) > 1:
#     #     if random.random() > 0.5:
#     #         continue

#     data_all.append(data)

for data_ind in range(num_data):
    with open(os.path.join(data_dir_2, f'{data_ind}.pkl'), 'rb') as f:
        data = pickle.load(f)
    data_all.append(data)

# sample
random.shuffle(data_all)
data_all = data_all[:num_data]

with open(output_path, 'wb') as f:
    pickle.dump(data_all, f)
print('Number of data:', len(data_all))