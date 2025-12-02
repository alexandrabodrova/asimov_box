""" Plot calibration curve after binning.

x-axis: confidence
y-axis: accuracy

"""
import os
import argparse
import pickle
from omegaconf import OmegaConf
from agent.predict.util import temperature_scaling
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(font_scale=1)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})


def main(cfg):

    # Load answer data
    data_path = os.path.join(cfg.parent_data_folder, cfg.data_path_from_parent)
    with open(data_path, 'rb') as f:
        data_all = pickle.load(f)

    num_bins = 10
    prob_bin_counts = [[] for _ in range(num_bins)]
    accuracy_bin_counts = [[] for _ in range(num_bins)]

    # Generate data
    for data_ind, data in enumerate(data_all):

        # Extract
        top_tokens = data['top_tokens']
        top_logprobs = data['top_logprobs']
        true_label = data['true_label']
        top_smx = temperature_scaling(top_logprobs, cfg.temperature_scaling)

        if len(true_label) > 1:
            continue

        # add each softmax to prob_bin_counts
        for top_smx_ind, top_smx_val in enumerate(top_smx):
            prob_bin_ind = int(top_smx_val * num_bins)
            prob_bin_counts[prob_bin_ind].append(top_smx_val)
            if top_tokens[top_smx_ind] in true_label:
                accuracy_bin_counts[prob_bin_ind].append(1)
            else:
                accuracy_bin_counts[prob_bin_ind].append(0)

    # Calculate accuracy
    accuracy = []
    confidence = []
    for prob_bin_ind in range(num_bins):
        if len(prob_bin_counts[prob_bin_ind]) > 0:
            accuracy.append(
                sum(accuracy_bin_counts[prob_bin_ind])
                / len(accuracy_bin_counts[prob_bin_ind])
            )
            confidence.append(
                sum(prob_bin_counts[prob_bin_ind])
                / len(prob_bin_counts[prob_bin_ind])
            )

    # Plot
    plt.plot(confidence, accuracy)
    # add diagonal dashed line
    plt.plot([0, 1], [0, 1], linestyle='--', color='r')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.savefig(
        os.path.join(cfg.parent_data_folder, cfg.save_fig_path_from_parent)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.data_folder = os.path.dirname(args.cfg_file)
    cfg.parent_data_folder = os.path.dirname(cfg.data_folder)

    # run
    main(cfg)