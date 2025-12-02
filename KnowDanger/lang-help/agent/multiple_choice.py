"""
Multiple choice agent.

"""
import random


class MultipleChoice:

    def __init__(self):
        pass

    def process_multiple_choice(self, response, add_mc=None):
        """
        Process multiple choice response from LM
        
        Args:
            response (str): response from LM
        
        Returns:
            mc_prompt (str): multiple choice prompt
            success (bool): whether the processing is successful
        """
        mc_all = response.split('\n')
        mc_processed_all = []
        for mc in mc_all:
            mc = mc.strip()  # sometimes there is leading space

            # skip nonsense
            if len(mc) < 5 or mc[0] not in [
                'a', 'b', 'c', 'd', 'A', 'B', 'C', 'D', '1', '2', '3', '4'
            ]:
                continue
            mc = mc[2:]  # remove a), b), ...
            mc = mc.strip().lower().split('.')[0]
            mc_processed_all.append(mc)
        if len(mc_processed_all) < 4:
            return '', False

        # Check if any repeated option
        mc_processed_all = list(set(mc_processed_all))
        if len(mc_processed_all) < 4:
            num_need = 4 - len(mc_processed_all)
            for _ in range(num_need):
                mc_processed_all.append('do nothing')

        prefix_all = ['A) ', 'B) ', 'C) ', 'D) ']
        if add_mc is not None:
            mc_processed_all.append(add_mc)
            prefix_all.append('E) ')
        random.shuffle(mc_processed_all)
        mc_prompt = ''
        for mc_ind, (prefix,
                     mc) in enumerate(zip(prefix_all, mc_processed_all)):
            mc_prompt += prefix + mc
            if mc_ind < len(mc_processed_all) - 1:
                mc_prompt += '\n'
        add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)
                                  ] if add_mc is not None else [-1]
        return mc_prompt, mc_processed_all, add_mc_prefix[0]
