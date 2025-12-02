"""
Base class of ambiguity.

"""
import itertools
import random
from string import Template


class BaseAmbiguity():
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
        self.numeric_choices = numeric_choices
        self.adj_choices = adj_choices
        self.obj_choices = obj_choices
        self.rel_choices = rel_choices
        self.action_choices = action_choices
        self.mc_template_choices = mc_template_choices
        self.object_set = cfg.object_set

        # Other cfg
        self.max_eq_mc = cfg.max_eq_mc
        self.max_amb_mc = cfg.max_amb_mc
        self.amb_mc_ratio = cfg.amb_mc_ratio
        self.num_mc_sample = cfg.num_mc_sample
        self.max_num_ambiguity_in_request = cfg.max_num_ambiguity_in_request

    def sample_object_set(self):
        random.shuffle(self.object_set)
        return self.object_set

    def sample_mc_templete(self):
        """Sample a multiple choice template."""
        return Template(random.choice(self.mc_template_choices))

    def generate_request(
        self, request_template, action, obj1, adj1, rel, obj2, adj2
    ):

        # Generate the request
        if rel is not None:
            rel_phrase = self.rel_choices[rel].request_phrase
        else:
            rel_phrase = 'on'  # arbitrary one if sem type
        request = request_template.substitute(
            action=action,
            adj1=adj1,
            obj1=obj1,
            rel=rel_phrase,
            adj2=adj2,
            obj2=obj2,
        )

        # remove space before comma in the request string
        request = request.replace(' ,', ',')
        # if action == 'put':
        #     action_cur = 'putting'
        # elif action == 'place':
        #     action_cur = 'placing'
        # elif action == 'move':
        #     action_cur = 'moving'
        # elif action == 'transfer':
        #     action_cur = 'transferring'
        # request = request.replace(action, action_cur)
        return request

    def sample_ground_truth_words(self, obj1, adj1, rel, obj2, adj2):
        """Sample ground truth from words in the request. If no equivalent or ambiguous words, return None to indicate the request is sem type."""

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
            return None, None, None, None, None
        else:
            return obj1_true, adj1_true, rel_true, obj2_true, adj2_true

    def get_rel_combinations(self, adj1, obj1, adj2, obj2):
        # To generate the multiple choices, we find all combinations of the adjective-object pairs that are equivalent, ambiguous, or semantically related.

        # Get eq combinations - both eq
        eq_combs = self.get_all_combinations(
            adj1,
            obj1,
            adj2,
            obj2,
            keys=['eq'],
            exclude_comb=[],
        )
        num_eq_mc = min(len(eq_combs), self.max_eq_mc)

        # Get all ambiguous combinations - both either eq or amb, but not both eq
        amb_combs = self.get_all_combinations(
            adj1,
            obj1,
            adj2,
            obj2,
            keys=['eq', 'amb'],
            exclude_comb=['eq eq eq eq'],
        )
        num_amb_mc = random.choices(
            range(self.max_amb_mc), weights=self.amb_mc_ratio, k=1
        )[0]
        num_amb_mc = min(len(amb_combs), self.max_amb_mc)

        # Get all semantic combinations - at least one of them is sem
        sem_combs = self.get_all_combinations(
            adj1,
            obj1,
            adj2,
            obj2,
            keys=['eq', 'amb', 'sem'],
            include_key='sem',
        )
        return eq_combs, amb_combs, sem_combs, num_eq_mc, num_amb_mc

    def get_all_combinations(
        self, adj1, obj1, adj2, obj2, keys, exclude_comb=[], include_key=''
    ):
        key_combs = list(itertools.product(keys, repeat=4))
        filtered_key_combs = []
        for key_comb in key_combs:
            format_key_comb = '{} {} {} {}'.format(
                key_comb[0], key_comb[1], key_comb[2], key_comb[3]
            )
            if format_key_comb in exclude_comb:
                continue
            if include_key not in format_key_comb:
                continue
            filtered_key_combs.append(format_key_comb)
        out = []
        for key_comb in filtered_key_combs:
            key1, key2, key3, key4 = key_comb.split(' ')
            combs = list(
                itertools.product(
                    sum([self.adj_choices[adj1][key1]], []),
                    sum([self.obj_choices[obj1][key2]], []),
                    sum([self.adj_choices[adj2][key3]], []),
                    sum([self.obj_choices[obj2][key4]], [])
                )
            )
            combs = [
                '{} {} {} {}'.format(x[0], x[1], x[2], x[3]) for x in combs
            ]
            out += combs
        return out
