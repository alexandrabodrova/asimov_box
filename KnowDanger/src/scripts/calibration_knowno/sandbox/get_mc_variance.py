att_data = 'data/vanilla_att/collect_mc/answer/answer.pkl'
num_data = 'data/vanilla_num/collect_mc/answer/answer.pkl'
spa_data = 'data/vanilla_spa/collect_mc/answer/answer.pkl'
saycan_data = 'data/saycan/collect/answer/answer.pkl'

import pickle
import numpy as np

with open(att_data, 'rb') as f:
    att_data = pickle.load(f)
with open(num_data, 'rb') as f:
    num_data = pickle.load(f)
with open(spa_data, 'rb') as f:
    spa_data = pickle.load(f)
with open(saycan_data, 'rb') as f:
    saycan_data = pickle.load(f)

att_var_all = []
for data in att_data:
    mc = data['mc_post']['mc_all']
    mc_word_len = [len(m.split()) for m in mc]
    mc_len_var = np.var(mc_word_len)
    att_var_all.append(mc_len_var)
att_var = np.mean(att_var_all)

num_var_all = []
for data in num_data:
    mc = data['mc_post']['mc_all']
    mc_word_len = [len(m.split()) for m in mc]
    mc_len_var = np.var(mc_word_len)
    num_var_all.append(mc_len_var)
num_var = np.mean(num_var_all)

spa_var_all = []
for data in spa_data:
    mc = data['mc_post']['mc_all']
    mc_word_len = [len(m.split()) for m in mc]
    mc_len_var = np.var(mc_word_len)
    spa_var_all.append(mc_len_var)
spa_var = np.mean(spa_var_all)

saycan_var_all = []
for data in saycan_data:
    mc = data['mc_all']
    mc_word_len = [len(m.split()) for m in mc]
    mc_len_var = np.var(mc_word_len)
    saycan_var_all.append(mc_len_var)
    breakpoint()
saycan_var = np.mean(saycan_var_all)

print(f'att_var: {att_var}')
print(f'num_var: {num_var}')
print(f'spa_var: {spa_var}')
print(f'saycan_var: {saycan_var}')