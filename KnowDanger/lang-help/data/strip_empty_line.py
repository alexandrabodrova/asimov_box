input_file_path = 'data/misc_test_palm/apr-26-pre.txt'
output_file_path = 'data/misc_test_palm/apr-26.txt'

with open(input_file_path, 'r') as f:
    data_all = f.read().split('--0000--')

# strip empty lines
data_all = [data.strip() for data in data_all]

# output
with open(output_file_path, 'w') as f:
    f.write('--0000--'.join(data_all))
print('Number of data: {}'.format(len(data_all)))