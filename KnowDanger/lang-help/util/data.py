import random
import logging


def process_answer_response(response):
    top_logprobs_full = response["choices"][0]["logprobs"]["top_logprobs"][0]
    top_token = response["choices"][0]["text"].strip()
    top_tokens = [token.strip().lower() for token in top_logprobs_full.keys()]
    top_logprobs = [value for value in top_logprobs_full.values()]
    return top_logprobs_full, top_token, top_tokens, top_logprobs


def determine_true_label_type(
    mc_types,
    mc_sigs,
    use_e_for_multiple_amb=False,
    use_e_for_possible_amb=False,
    num_amb_mc_possible=None,
):
    """ For ambiguous type, choose 'e/E' if there is more than one ambiguous choice."""
    if len(mc_sigs) == 5:
        e_sig = mc_sigs[-1]

    if 'eq' in mc_types:
        eq_indices = [i for i, x in enumerate(mc_types) if x == 'eq']
        eq_index = random.choice(eq_indices)
        true_label = mc_sigs[eq_index]
        true_type = 'eq'
    elif 'amb' in mc_types:
        amb_indices = [i for i, x in enumerate(mc_types) if x == 'amb']
        if use_e_for_multiple_amb and len(amb_indices) > 1:
            true_label = e_sig
        else:
            # if there exists a different amb action not covered by the four options, include e as possible true label too, if use_e_for_possible_amb is set to True
            if num_amb_mc_possible is not None:
                assert len(amb_indices) <= num_amb_mc_possible
                mc_labels_tmp = [mc_sigs[i] for i in amb_indices]
                if len(
                    amb_indices
                ) != num_amb_mc_possible and use_e_for_possible_amb:
                    mc_labels_tmp.append(e_sig)
            else:
                mc_labels_tmp = [mc_sigs[i] for i in amb_indices]
                if use_e_for_possible_amb:
                    mc_labels_tmp.append(e_sig)
            true_label = random.choice(mc_labels_tmp)
            # true_label = mc_labels_tmp[amb_index]
        true_type = 'amb'
    else:
        true_label = e_sig
        true_type = 'sem'
    return true_label, true_type


def postprocess_mc(
    mc_all,
    mc_types,
    mc_sigs=['A', 'B', 'C', 'D', 'E'],
    add_mc='an option not listed here',
    verbose=True,
):
    """Shuffle the multiple choices, prepend the letter, and log them"""
    if add_mc is not None:
        mc_all.append(add_mc)
        mc_types.append('sem')

    shuffle_order = [i for i in range(len(mc_all))]
    random.shuffle(shuffle_order)
    mc_all = [mc_all[i] for i in shuffle_order]
    mc_types = [mc_types[i] for i in shuffle_order]
    mc_prompt = ''
    for sig, mc_type, mc in zip(mc_sigs, mc_types, mc_all):
        if verbose:
            logging.info(f'{sig}) {mc} - {mc_type}')
        mc_prompt += f'{sig}) {mc}\n'

    mc_prompt = mc_prompt.strip()
    return mc_prompt, mc_all, mc_types
