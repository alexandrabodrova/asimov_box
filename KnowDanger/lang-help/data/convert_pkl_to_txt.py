import pickle


input_file_path = 'data/v4_clarify_binary_ms/collect/step_1_clarify.pkl'
output_file_path = 'data/v4_clarify_binary_ms/collect/step_1_clarify.txt'

with open(input_file_path, 'rb') as f:
    data_all = pickle.load(f)

# strip empty lines
data_all = [data['prompt'].strip() for data in data_all]

# output with '--0000--' as delimiter
with open(output_file_path, 'w') as f:
    f.write('--0000--'.join(data_all))
