"""
Read the PaLM prompt and response together. Mostly for experimenting with PaLM.

"""

prompt_data_path = 'data/misc_test_palm/apr-26.txt'
response_data_path = 'data/misc_test_palm/apr-26-res.txt'

# Load prompt data
with open(prompt_data_path, 'r') as f:
    prompt_data_all = f.read().split('--0000--')

# Load response data
with open(response_data_path, 'r') as f:
    response_data_all = f.read().split('--0000--')
assert len(prompt_data_all) == len(response_data_all)

# Read prompt and response together
for prompt_data, response_data in zip(prompt_data_all, response_data_all):
    print("==================")
    print(prompt_data)
    print("(prompt end)")
    print(response_data)
    print("==================")

    input('Press any key to continue...')