"""
Plot the softmax distribution of all options across data. Better understand the distribution and the effect of temperature scaling.

"""
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


sns.set(font_scale=2.5)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})

from agent.predict.util import temperature_scaling


SMX_CUTOFF = 0.01


def main(args, mc_sigs):
    with open(args.data_path, 'rb') as f:
        data_all = pickle.load(f)

    logprobs_token_all = {sig: [] for sig in mc_sigs}
    smx_token_all = {sig: [] for sig in mc_sigs}
    smx_token_scaled_all = {sig: [] for sig in mc_sigs}
    for data in data_all:
        # top_logprobs_full = data['lm_response']["choices"][0]["logprobs"][
        #     "top_logprobs"][0]
        # top_tokens = [token.strip() for token in top_logprobs_full.keys()]
        # top_logprobs = [value for value in top_logprobs_full.values()]
        # true_label = data['true_label']
        top_logprobs = data['top_logprobs']
        top_tokens = data['top_tokens']

        # temperature scaling and get softmax
        smx_scaled = temperature_scaling(top_logprobs, args.temperature)

        # Save
        for ind, token in enumerate(top_tokens):
            logprobs_token_all[token].append(top_logprobs[ind])
            if np.exp(top_logprobs[ind]) > SMX_CUTOFF and np.exp(
                top_logprobs[ind]
            ) < 1 - SMX_CUTOFF:
                smx_token_all[token].append(np.exp(top_logprobs[ind]))
                smx_token_scaled_all[token].append(smx_scaled[ind])

    # plot histogram for log prob of each a/b/c/d/e
    fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(15, 15))
    for ind, key in enumerate(logprobs_token_all):
        axs[ind].hist(logprobs_token_all[key], bins=50, alpha=0.5)
        axs[ind].set_title(key)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False
    )
    plt.xlabel("log probability raw")
    plt.ylabel("count")
    # plt.show()
    plt.savefig(os.path.join(args.save_dir, 'logprob_raw.png'))

    # plot histogram for softmax of each a/b/c/d/e
    fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(15, 15))
    for ind, key in enumerate(smx_token_all):
        axs[ind].hist(smx_token_all[key], bins=50, alpha=0.5)
        axs[ind].set_title(key)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False
    )
    plt.xlabel("softmax raw")
    plt.ylabel("count")
    # plt.show()
    plt.savefig(os.path.join(args.save_dir, 'softmax_raw.png'))

    # plot histogram for softmax of each a/b/c/d/e (temperature scaled)
    fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(15, 15))
    for ind, key in enumerate(smx_token_scaled_all):
        axs[ind].hist(smx_token_scaled_all[key], bins=50, alpha=0.5)
        axs[ind].set_title(key)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor='none', top=False, bottom=False, left=False, right=False
    )
    plt.xlabel("softmax scaled")
    plt.ylabel("count")
    # plt.show()
    plt.savefig(os.path.join(args.save_dir, 'softmax_scaled.png'))

    # plot all logprob together
    fig = plt.figure(figsize=(15, 15))
    plt.hist(sum(logprobs_token_all.values(), []), bins=100, alpha=0.5)
    plt.xlabel("log probability raw")
    plt.ylabel("count")
    plt.title("All options")
    # plt.show()
    plt.savefig(os.path.join(args.save_dir, 'logprob_raw_all.png'))

    # plot all softmax together
    fig = plt.figure(figsize=(15, 15))
    plt.hist(sum(smx_token_all.values(), []), bins=100, alpha=0.5)
    plt.xlabel("softmax raw")
    plt.ylabel("count")
    plt.title("All options")
    # plt.show()
    plt.savefig(os.path.join(args.save_dir, 'softmax_raw_all.png'))

    # plot all softmax together (temperature scaled)
    fig = plt.figure(figsize=(15, 15))
    plt.hist(sum(smx_token_scaled_all.values(), []), bins=100, alpha=0.5)
    plt.xlabel("softmax scaled")
    plt.ylabel("count")
    plt.title("All options (temperature scaled)")
    # plt.show()
    plt.savefig(os.path.join(args.save_dir, 'softmax_scaled_all.png'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=
        '/home/allen/lang-help/data/saycan/collect/answer_002/answer.pkl',
        help="Path to data file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Temperature for temperature scaling",
    )
    args = parser.parse_args()
    args.save_dir = os.path.dirname(args.data_path)
    main(args, mc_sigs=['A', 'B', 'C', 'D', 'E'])
