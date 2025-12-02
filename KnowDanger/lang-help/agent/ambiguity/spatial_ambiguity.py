"""
Spatial type ambiguity.

"""
import random
import logging
import itertools

from .base_ambiguity import BaseAmbiguity


class SpatialAmbiguity(BaseAmbiguity):
    """
    Args:
        cfg (dict): configuration
        adj_choices (dict): adjectives and related words
        obj_choices (dict): objects and related words
        rel_choices (dict): relations and related words
        action_choices (dict): actions and related words
        mc_template_choices (list): templates of multiple choice
    """

    def __init__(
        self,
        cfg,
        numeric_choices,
        adj_choices,
        obj_choices,
        rel_choices,
        action_choices,
        mc_template_choices,
    ):
        super().__init__(
            cfg,
            numeric_choices,
            adj_choices,
            obj_choices,
            rel_choices,
            action_choices,
            mc_template_choices,
        )
        self.sptial_ambiguous_rel_choices = {
            key: value for key, value in self.rel_choices.items()
            # if value['spatial_ambiguous']
        }

    def sample_request_setting(self):
        while 1:
            action = random.choice(self.action_choices)
            rel = random.choice(list(self.sptial_ambiguous_rel_choices.keys()))
            objs = random.sample(list(self.obj_choices.keys()), k=2)
            obj1, obj2 = objs
            adjs = random.sample(list(self.adj_choices.keys()), k=2)
            adj1, adj2 = adjs

            # Make sure the two objects are different
            adj1_eq_amb = self.adj_choices[adj1].eq + self.adj_choices[adj1].amb
            obj1_eq_amb = self.obj_choices[obj1].eq + self.obj_choices[obj1].amb
            adj1_obj1_combs = list(itertools.product(adj1_eq_amb, obj1_eq_amb))
            adj2_eq_amb = self.adj_choices[adj2].eq + self.adj_choices[adj2].amb
            obj2_eq_amb = self.obj_choices[obj2].eq + self.obj_choices[obj2].amb
            adj2_obj2_combs = list(itertools.product(adj2_eq_amb, obj2_eq_amb))
            if len(
                set(adj1_obj1_combs).intersection(set(adj2_obj2_combs))
            ) > 0:
                continue

            # Make sure there is no more than max number of ambiguities in the request - relation is always ambiguous, thus look at adjectives and objects
            num_ambiguity_in_request = 1
            if not self.adj_choices[adj1].eq and self.adj_choices[adj1].amb:
                num_ambiguity_in_request += 1
            if not self.adj_choices[adj2].eq and self.adj_choices[adj2].amb:
                num_ambiguity_in_request += 1
            if not self.obj_choices[obj1].eq and self.obj_choices[obj1].amb:
                num_ambiguity_in_request += 1
            if not self.obj_choices[obj2].eq and self.obj_choices[obj2].amb:
                num_ambiguity_in_request += 1
            if num_ambiguity_in_request > self.max_num_ambiguity_in_request:
                continue
            break
        flag_ambiguous = num_ambiguity_in_request > 1
        return action, obj1, adj1, rel, obj2, adj2, flag_ambiguous

    def generate_mc(self, obj1, adj1, rel, obj2, adj2, exclude_sem=False):
        """
        Difference from attribute: here when we sample the relation, we always sample a sem phrase related to it, which is in the format of get_obs_pos() + [X, Y]'. Then we fill in the pos part of the multiple choice template.

        In this case, the true answer is always None of the above.
        
        We still sample the attribute part with the same rule as in Attribute.
        """

        # Get relation combinations
        eq_combs, amb_combs, sem_combs, num_eq_mc, num_amb_mc = self.get_rel_combinations(
            adj1, obj1, adj2, obj2
        )
        # logging.info(f'eq comb: {eq_combs}')
        # logging.info(f'amb comb: {amb_combs}')

        # Check if any combination exists
        flag_eq_exist = len(eq_combs) > 0
        flag_amb_exist = len(amb_combs) > 0
        flag_sem_exist = len(sem_combs) > 0
        if not flag_eq_exist and not flag_amb_exist and not flag_sem_exist:
            print('No combination exists!\n')
            return None
        elif num_eq_mc + num_amb_mc + len(sem_combs) < self.num_mc_sample:
            print('Not enough choices!\n')
            return None
        # elif exclude_sem and (not flag_amb_exist and not flag_eq_exist):
        #     print('Sem target reached and no amb/eq mc exists!\n')
        #     return None

        # Generate all choices
        mc_all = []
        mc_types = []
        comb_seen = []  # avoid repeated mc
        num_amb_mc_tmp = num_amb_mc
        while len(mc_all) < self.num_mc_sample:

            # Sample the template
            mc_template = self.sample_mc_templete()

            # Sample combination
            if flag_eq_exist:  # sample equivalent until maximum number reached
                comb12 = random.choice(eq_combs)
                if comb12 in comb_seen:
                    continue
                comb_seen.append(comb12)
                mc_types.append('eq')

                # Count
                num_eq_mc -= 1
                if num_eq_mc == 0:
                    flag_eq_exist = False

            elif flag_amb_exist:  # sample ambiguous until maximum number reached, or if no semantic combination exists
                comb12 = random.choice(amb_combs)
                if comb12 in comb_seen:
                    continue
                comb_seen.append(comb12)
                mc_types.append('amb')

                # Count
                num_amb_mc_tmp -= 1
                if num_amb_mc_tmp == 0:
                    flag_amb_exist = False

            else:  # sample semantic
                comb12 = random.choice(sem_combs)
                if comb12 in comb_seen:
                    continue
                comb_seen.append(comb12)
                mc_types.append('sem')

            # Split the combination into words
            mc_adj1, mc_obj1, mc_adj2, mc_obj2 = comb12.split(' ')

            # Sample the action
            mc_action = random.choice(self.action_choices)

            # Sample the spatial relation (always amb) - downgrade to amb if eq
            mc_rel = random.choice(
                self.sptial_ambiguous_rel_choices[rel]['amb']
            )
            if mc_types[-1] == 'eq':
                mc_types[-1] = 'amb'

            # actual rel can be from any rel
            mc_rel_phrase = self.rel_choices[mc_rel].mc_phrase
            mc_rel_phrase = mc_rel_phrase.replace('adj2', mc_adj2
                                                 ).replace('obj2', mc_obj2)

            # Fill in template
            mc = mc_template.substitute(
                action=mc_action,
                adj1=mc_adj1,
                obj1=mc_obj1,
                rel_phrase=mc_rel_phrase,
            )
            mc_all.append(mc)

        # Return mc and info
        info = {
            'num_amb_mc': num_amb_mc,
            'num_amb_mc_possible': 100
        }  # make a big number
        return mc_all, mc_types, info
