import numpy as np
import random


def get_score(
    top_tokens,
    top_smx,
    true_label,
    score_method='conformal',
    cfg=None,
):
    """Get score function based on score method."""

    # Set RAPS regularization parameters (larger lam_reg and smaller k_reg leads to smaller sets)
    if score_method == 'regularized_adaptive_conformal':
        num_classes = len(top_tokens)
        reg_vec = np.array(
            cfg.k_reg * [
                0,
            ] + (num_classes - cfg.k_reg) * [
                cfg.lam_reg,
            ]
        )

    if score_method == 'conformal':  # 1 - softmax(true label)
        cal_score = 1 - top_smx[top_tokens.index(true_label)]
    if score_method == 'conformal_top_k':
        top_smx_argsort_des = top_smx.argsort()[::-1]
        top_tokens_sorted = [top_tokens[i] for i in top_smx_argsort_des]
        cal_score = top_tokens_sorted.index(true_label) + 1
    elif score_method == 'adaptive_conformal':  # cumsum until true label
        top_smx_argsort_des = top_smx.argsort()[::-1]
        top_tokens_sorted = [top_tokens[i] for i in top_smx_argsort_des]
        top_smx_sorted_cumsum = top_smx[top_smx_argsort_des].cumsum()
        cal_score = top_smx_sorted_cumsum[top_tokens_sorted.index(true_label)]
    elif score_method == 'regularized_adaptive_conformal':
        top_smx_argsort_des = top_smx.argsort()[::-1]
        top_tokens_sorted = [top_tokens[i] for i in top_smx_argsort_des]
        top_smx_sorted = np.take_along_axis(
            top_smx, top_smx_argsort_des, axis=0
        )
        top_smx_sorted_reg = top_smx_sorted + reg_vec
        cal_L = top_tokens_sorted.index(true_label)
        cal_score = top_smx_sorted_reg.cumsum(
        )[cal_L] - random.random() * top_smx_sorted_reg[cal_L]
    elif score_method == 'naive':
        cal_score = 0  # dummy

    return cal_score


def get_prediction_set(
    top_tokens,
    top_smx,
    qhat=None,
    score_method='conformal',
    cfg=None,
):
    """Get the prediction set based on score method and qhat (quantile)."""

    # Set RAPS regularization parameters (larger lam_reg and smaller k_reg leads to smaller sets)
    if score_method == 'regularized_adaptive_conformal':
        num_classes = len(top_tokens)
        reg_vec = np.array(
            cfg.k_reg * [
                0,
            ] + (num_classes - cfg.k_reg) * [
                cfg.lam_reg,
            ]
        )

    # Get prediction set
    if score_method == 'conformal':  # include all choices with softmax score >= 1-qhat
        prediction_set = [
            token for token_ind, token in enumerate(top_tokens)
            if top_smx[token_ind] >= 1 - qhat
        ]
    elif score_method == 'conformal_top_k':
        top_smx_argsort_des = top_smx.argsort()[::-1]
        top_tokens_sorted = [top_tokens[i] for i in top_smx_argsort_des]
        prediction_set = top_tokens_sorted[:int(np.ceil(qhat))]
    elif score_method == 'adaptive_conformal':  # include all choices with cumsum softmax score >= qhat
        top_smx_argsort_des = top_smx.argsort()[::-1]
        top_tokens_sorted = [top_tokens[i] for i in top_smx_argsort_des]
        top_smx_sorted_cumsum = top_smx[top_smx_argsort_des].cumsum()
        cumsum_index = np.argmax(top_smx_sorted_cumsum >= qhat)
        prediction_set = top_tokens_sorted[:cumsum_index + 1]
    elif score_method == 'regularized_adaptive_conformal':
        top_smx_argsort_des = top_smx.argsort()[::-1]
        top_tokens_sorted = [top_tokens[i] for i in top_smx_argsort_des]
        top_smx_sorted = np.take_along_axis(
            top_smx, top_smx_argsort_des, axis=0
        )
        top_smx_sorted_reg = top_smx_sorted + reg_vec
        top_smx_sorted_reg_cumsum = top_smx_sorted_reg.cumsum()
        indicators = (
            top_smx_sorted_reg_cumsum - random.random() * top_smx_sorted_reg
        ) <= qhat if cfg.rand else top_smx_sorted_reg_cumsum - top_smx_sorted_reg <= qhat
        if cfg.disallow_zero_sets:
            indicators[0] = True
        prediction_set = [
            top_tokens_sorted[i]
            for i, indicator in enumerate(indicators)
            if indicator
        ]
    elif score_method == 'naive':
        top_smx_argsort_des = top_smx.argsort()[::-1]
        top_tokens_sorted = [top_tokens[i] for i in top_smx_argsort_des]
        top_smx_sorted_cumsum = top_smx[top_smx_argsort_des].cumsum()
        if top_smx_sorted_cumsum[
            -1
        ] < cfg.naive_cal_level:  # if top five tokens summed logprob is still lower than threshold
            prediction_set = top_tokens_sorted
        else:
            prediction_set = top_tokens_sorted[:np.argmax(
                top_smx_sorted_cumsum >= cfg.naive_cal_level
            ) + 1]

    return prediction_set


def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx
