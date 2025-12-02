import pickle
import os
import random


data_dir = 'data/saycan_final/collect/answer/single_data'
output_path = 'data/saycan_final/collect/answer/answer.pkl'
num_data = 800

data_all = []
for data_ind in range(num_data):
    with open(os.path.join(data_dir, f'{data_ind}.pkl'), 'rb') as f:
        data = pickle.load(f)

    true_label = data['true_label']
    if len(
        true_label[0]
    ) == 0:  # check if accidentally put multiple latters for a label
        print(data['request'])
        print(data['mc_prompt'])
        print(data['true_intent'])

    # print(data['task_category'])
    # if data['task_category'] == 'unambiguous_task':
    #     if random.random() > 0.5:
    #         continue
    # if len(true_label) > 1:
    #     continue
    # if len(true_label) > 1:
    #     if random.random() > 0.5:
    #         continue

    data_all.append(data)

with open(output_path, 'wb') as f:
    pickle.dump(data_all, f)
print('Number of data:', len(data_all))