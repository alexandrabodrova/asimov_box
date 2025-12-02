import pickle


combined_data_path = 'data/v1_easy_122822/combined_data.pkl'
with open(combined_data_path, 'rb') as handle:
    combined_data_all = pickle.load(handle)

split = [0, 0, 0]
bad_split = [0, 0, 0]
for data in combined_data_all:
    context = data['context']
    true_label = data['true_label']
    true_type = data['true_type']
    response_full = data['lm_response']

    # Extract top tokens and logprobs
    top_logprobs_full = response_full["choices"][0]["logprobs"]["top_logprobs"
                                                               ][0]
    top_token = response_full["choices"][0]["text"].strip()
    top_tokens = [token.strip().lower() for token in top_logprobs_full.keys()]
    top_logprobs = [value for value in top_logprobs_full.values()]

    if true_type == 'eq':
        split[0] += 1
    elif true_type == 'amb':
        split[1] += 1
    elif true_type == 'sem':
        split[2] += 1

    # debug
    if true_label in top_tokens:
        true_logprob_ind = top_tokens.index(true_label)
        true_logprob = top_logprobs[true_logprob_ind]
        if true_logprob < -10:
            if true_type == 'eq':
                bad_split[0] += 1
            elif true_type == 'amb':
                bad_split[1] += 1
            elif true_type == 'sem':
                bad_split[2] += 1
            print(context)
            print(true_label, true_type)
            print(top_logprobs_full)
            print()

print(split)
print(bad_split)
print(
    'Bad ratio: ', bad_split[0] / split[0], bad_split[1] / split[1],
    bad_split[2] / split[2]
)
