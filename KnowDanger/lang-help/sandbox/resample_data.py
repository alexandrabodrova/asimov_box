import pickle
import os
import random


data_dir = 'data/saycan_palm/collect/answer_no_probe/single_data'
output_path = 'data/saycan_palm/collect/answer_no_probe/answer_3.pkl'
num_data = 700

old_data_all = []
task_category_all = {}
# ambiguous_task, creative_unambiguous_task, spatial_ambiguous_task, unambiguous_task, creative_ambiguous_task, unsafe_task
task_category_weight = [0.15, 0.15, 0.3, 0.1, 0.15, 0.15]

for data_ind in range(num_data):
    with open(os.path.join(data_dir, f'{data_ind}.pkl'), 'rb') as f:
        data = pickle.load(f)

    true_label = data['true_label']
    task_category = data['task_category']

    if task_category not in task_category_all:
        task_category_all[task_category] = []
    task_category_all[task_category].append(data_ind)

    old_data_all.append(data)
    # print(data['task_category'])

    # if data['task_category'] == 'unambiguous_task':
    #     if random.random() > 0.5:
    #         continue
    # if len(true_label) > 1:
    #     continue
    # if len(true_label) > 1:
    #     if random.random() > 0.5:
    #         continue

new_data_all = []
task_category_names = list(task_category_all.keys())
for data_ind in range(num_data):
    task_category = random.choices(
        task_category_names, weights=task_category_weight
    )[0]
    data_ind = random.choice(task_category_all[task_category])
    new_data_all.append(old_data_all[data_ind])
    print(task_category)

# print(task_category_all)

with open(output_path, 'wb') as f:
    pickle.dump(new_data_all, f)
print('Number of data:', len(new_data_all))