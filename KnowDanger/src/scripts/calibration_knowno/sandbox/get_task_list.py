import pickle

# init_data_path = 'data/saycan/collect/answer/answer.pkl'
# # load
# with open(init_data_path, 'rb') as f:
#     init_data = pickle.load(f)

# request_all = []
# scene_objects_all = []
# for data in init_data:
#     try:
#         request = data['request']
#     except:
#         request = data['task_prompt']
#     scene_objects = data['scene_objects']
#     if request not in request_all:
#         request_all.append(request)
#         scene_objects_all.append(scene_objects)

# # print to text file
# with open('saycan_tasks.txt', 'w') as f:
#     for i in range(len(request_all)):
#         f.write('Task: ' + request_all[i] + '\n')
#         # print objects in one line
#         f.write('Objects: ')
#         for j in range(len(scene_objects_all[i])):
#             f.write(scene_objects_all[i][j])
#             if j != len(scene_objects_all[i]) - 1:
#                 f.write(', ')
#         f.write('\n\n')

init_data_path = 'data/bimanual/collect/answer/answer.pkl'
# load
with open(init_data_path, 'rb') as f:
    init_data = pickle.load(f)

request_all = []
for data in init_data:
    request = data['request']
    if request not in request_all:
        request_all.append(request)

# print to text file
with open('bimanual_tasks.txt', 'w') as f:
    for i in range(len(request_all)):
        f.write('Task: ' + request_all[i] + '\n')
        f.write('\n')