"""
Syntatic type ambiguity. Example: "second blue object"

"""
import random
import logging
from .base_ambiguity import BaseAmbiguity


class SyntacticAmbiguity(BaseAmbiguity):
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

        # cfg
        self.rel_eq_ratio = cfg.attribute_rel_eq_ratio

        # only use unambiguous relations
        self.rel_choices = {
            key: value
            for key, value in self.rel_choices.items()
            if not value.spatial_ambiguous
        }

        # True attributes and objects
        self.true_obj_choices = ['bowl', 'block']
        self.true_adj_choices = ['blue', 'yellow', 'green']

    def sample_object_set(self):
        """Use ones sampled in request setting, without shuffling"""
        return self.object_set

    def sample_request_setting(self):
        while 1:
            action = random.choice(self.action_choices)
            rel = random.choice(list(self.rel_choices.keys()))
            obj2 = random.choice(self.true_obj_choices)

            # a bit hacky to figure out adj1
            adj_1_index = random.choice([0, 1])
            adj_index_words = ['first', 'second']
            adj_1_index_word = adj_index_words[adj_1_index]
            adj1 = random.choice(self.true_adj_choices)
            adj1_full = adj_1_index_word + ' ' + adj1
            adj2 = random.choice(self.true_adj_choices)

            # make sure the two objects are different
            random.shuffle(self.object_set)
            # get the adj_1_index-th occurrence of adj1_adj in object_set
            adj_obj_1 = [
                obj for obj in self.object_set if obj.startswith(adj1)
            ][adj_1_index]
            if adj_obj_1 != adj2 + ' ' + obj2:
                break

        # Save the actual adj1 and obj1 for latyer use
        obj1 = 'object'
        self.true_adj1 = adj1
        self.true_obj1 = adj_obj_1.split(' ')[1]

        # randomly choose to use syntatic on obj1 or obj2
        self.use_syntactic_obj1 = random.choice([True, False])

        flag_ambiguous = True  # TODO: needs to better define this
        if self.use_syntactic_obj1:
            return action, obj1, adj1_full, rel, obj2, adj2, flag_ambiguous
        else:
            return action, obj2, adj2, rel, obj1, adj1_full, flag_ambiguous

    # override
    def sample_ground_truth_words(self, obj1, adj1, rel, obj2, adj2):
        """Sample ground truth from words in the request. If no equivalent or ambiguous words, return None to indicate the request is sem type."""

        # swap obj1 and obj2 if not use_syntactic_obj1
        if not self.use_syntactic_obj1:
            obj1, obj2 = obj2, obj1
            adj1, adj2 = adj2, adj1

        # get the actual adj1 and obj1
        adj1 = self.true_adj1
        obj1 = self.true_obj1

        def find_eq_or_amb(word, word_type):
            if word_type == 'obj':
                choices = self.obj_choices
            elif word_type == 'adj':  # yikes
                if word in self.adj_choices.keys():
                    choices = self.adj_choices
                else:
                    choices = self.numeric_choices
            elif word_type == 'rel':
                choices = self.rel_choices
            if len(choices[word]['eq']
                  ) == 0 and len(choices[word]['amb']) == 0:
                return None
            else:
                return random.choice(
                    choices[word]['eq'] + choices[word]['amb']
                )

        obj1_true = find_eq_or_amb(obj1, 'obj')
        adj1_true = find_eq_or_amb(adj1, 'adj')
        rel_true = find_eq_or_amb(rel, 'rel')
        obj2_true = find_eq_or_amb(obj2, 'obj')
        adj2_true = find_eq_or_amb(adj2, 'adj')

        # return all None of any of them is None
        if obj1_true is None or adj1_true is None or rel_true is None or obj2_true is None or adj2_true is None:
            return None
        elif self.use_syntactic_obj1:
            return obj1_true, adj1_true, rel_true, obj2_true, adj2_true
        else:
            return obj2_true, adj2_true, rel_true, obj1_true, adj1_true

    def generate_mc(self, obj1, adj1, rel, obj2, adj2, exclude_sem=False):

        # get the actual adj1 and obj1
        adj1 = self.true_adj1
        obj1 = self.true_obj1

        # swap obj1 and obj2 if not use_syntactic_obj1
        if not self.use_syntactic_obj1:
            obj1, obj2 = obj2, obj1
            adj1, adj2 = adj2, adj1

        # Get relation combinations
        eq_combs, amb_combs, sem_combs, num_eq_mc, num_amb_mc = self.get_rel_combinations(
            adj1,
            obj1,
            adj2,
            obj2,
        )

        # Check if any combination exists
        flag_eq_exist = len(eq_combs) > 0
        flag_amb_exist = len(amb_combs) > 0
        flag_sem_exist = len(sem_combs) > 0
        if not flag_eq_exist and not flag_amb_exist and not flag_sem_exist:
            logging.info('No combination exists!\n')
            return None
        elif num_eq_mc + num_amb_mc + len(sem_combs) < self.num_mc_sample:
            logging.info('Not enough choices!\n')
            return None

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

            # Determine if using correct relation
            flag_rel_eq = random.random() < self.rel_eq_ratio

            # Sample the spatial relation
            if flag_rel_eq or self.rel_choices[rel]['sem'] == []:
                mc_rel = random.choice(self.rel_choices[rel]['eq'])
            else:
                mc_rel = random.choice(self.rel_choices[rel]['sem'])
                mc_types[-1] = 'sem'  # override the type

            # Fill in rel
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
            'num_amb_mc_possible': len(amb_combs)
        }
        return mc_all, mc_types, info
