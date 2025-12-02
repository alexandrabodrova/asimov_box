"""Numeric type ambiguity.

Example:
    request: put some blocks into the green bowl
    choices:
        a) put the yellow block in the green bowl, put the blue block in the green bowl
        b) put the blue block in the green bowl
        c) put the green block in the green bowl, put the yellow block in the green bowl, put the blue block in the green bowl
        d) put the green block in the green bowl, put the blue block in the green bowl
        e) you are not sure
    correct answer: e), since the number of blocks to be moved is ambiguous.

We always use true attributes and objects here, such as yellow block or green bowl, in the multiple choices.

Multiple choices can be repeated easily. TODO: fix this.
"""
import random

from .base_ambiguity import BaseAmbiguity


MC_PREFIX = ['first, ', 'second, ', 'third, ']
MC_NUM_CONVERSION = {'one': 1, 'two': 2, 'three': 3}


class NumericAmbiguity(BaseAmbiguity):
    """
    Args:
        cfg (dict): configuration
        numeric_choices (dict): number of objects
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
        self.max_num_phrases = cfg.numeric_max_num_phrases

        # only use non-ambiguous relations
        self.rel_choices = {
            key: value
            for key, value in self.rel_choices.items()
            if not value.spatial_ambiguous
        }

        # cfg
        self.correct_ratio = cfg.numeric_correct_ratio

        # True attributes and objects
        self.true_obj_choices = list(
            key for key in self.obj_choices.keys()
            if len(self.obj_choices[key]['eq']) > 0
        )
        self.true_adj_choices = list(
            key for key in self.adj_choices.keys()
            if len(self.adj_choices[key]['eq']) > 0
        )

        # # substitute true_adj_choices with eq
        # self.true_adj_choices = [
        #     self.adj_choices[adj]['eq'][0] for adj in self.true_adj_choices
        # ]
        # self.true_adj_choices = list(set(self.true_adj_choices))

    def sample_request_setting(self):
        action = random.choice(self.action_choices)
        rel = random.choice(list(self.rel_choices.keys()))
        while 1:
            objs = random.sample(self.true_obj_choices, k=2)
            obj1, obj2 = objs
            # make sure not sampling block and block, or cube and block
            # e.g., "put all cubes on the yellow block"
            # probably relax this later
            if obj1 not in self.obj_choices[obj2][
                'eq'] and obj2 not in self.obj_choices[obj1]['eq']:
                break
        num = random.choices(
            list(self.numeric_choices.keys()),
            weights=[cfg.weight for cfg in self.numeric_choices.values()]
        )[0]  #! weight
        adj2 = random.choice(self.true_adj_choices)
        flag_ambiguous = self.numeric_choices[num]['eq'] == []
        return action, obj1, num, rel, obj2, adj2, flag_ambiguous

    def generate_request(
        self, request_template, action, obj1, num, rel, obj2, adj2
    ):
        request = super().generate_request(
            request_template, action, obj1, num, rel, obj2, adj2
        )

        # Remove the first the last the from context (e.g., "put the all block on the blue block" -> "put all block on the blue block")
        request = request.replace('the ', '', 1)

        # Make the obj1 plural except for eq=1
        if not (
            len(self.numeric_choices[num]['eq']) > 0
            and MC_NUM_CONVERSION[self.numeric_choices[num]['eq'][0]] == 1
        ):
            plural = self.obj_choices[obj1]['plural']
            request = request.replace(obj1, plural)
        return request

    def generate_mc(self, obj1, num, rel, obj2, adj2, exclude_sem=False):

        # Check if sem target reached
        # if exclude_sem:
        #     logging.info('Sem target reached and no amb/eq mc exists!\n')
        #     return None

        # Substitute obj1, obj2, adj2 with eq
        obj1 = random.choice(self.obj_choices[obj1]['eq'])
        obj2 = random.choice(self.obj_choices[obj2]['eq'])
        adj2 = random.choice(self.adj_choices[adj2]['eq'])

        # Check if numeric relation exists - assume no amb right now
        flag_numeric_eq_exist = len(self.numeric_choices[num]['eq']) > 0

        # Whether to include the correct one in the mc
        flag_use_correct = random.random() < self.correct_ratio

        # Generate all choices
        mc_all = []
        mc_types = []
        while len(mc_all) < self.num_mc_sample:

            # Initialize mc
            mc = ''

            # Sample the template
            mc_template = self.sample_mc_templete()

            # Sample number of phrases
            if flag_numeric_eq_exist and flag_use_correct:
                mc_num = MC_NUM_CONVERSION[random.choice(
                    self.numeric_choices[num]['eq']
                )]
                flag_numeric_eq_exist = False
                mc_types.append('eq')
            else:
                mc_num = MC_NUM_CONVERSION[random.choice(
                    self.numeric_choices[num]['sem']
                )]
                mc_types.append('sem')

            # Generate each phrase
            true_attribute_samples = random.sample(
                self.true_adj_choices,
                k=mc_num,
            )
            # for i in range(self.max_num_phrases):

            #     if i < mc_num:
            #         # Still substitute with eq attribute, so multiple choice only contains recognizable words
            #         attribute = true_attribute_samples[i]
            #         # attribute = random.choice(self.adj_choices[attribute]['eq'])

            #         mc_action = random.choice(self.action_choices)

            #         # use the exactly same adj and obj for the phrase
            #         mc_rel = random.choice(self.rel_choices[rel]['eq']).replace('adj2', adj2).replace('obj2', obj2)

            #         mc_tmp = mc_template.substitute(action=mc_action,
            #                                     adj1=attribute,
            #                                     obj1=obj1,
            #                                     rel_phrase=mc_rel)

            #         # add first, second, third to front
            #         mc += MC_PREFIX[i] + mc_tmp + '; '

            #     # add do nothing
            #     else:
            #         mc += MC_PREFIX[i] + 'do nothing'
            #         if i != self.max_num_phrases - 1:
            #             mc += '; '

            # use a single phrase instead
            obj1_phrase = ''
            for i in range(mc_num):
                attribute = true_attribute_samples[i]
                if i > 0:
                    attribute = 'the ' + attribute
                obj1_phrase += attribute + ' ' + obj1
                if i != mc_num - 1:
                    obj1_phrase += ' and '

            mc_action = random.choice(self.action_choices)

            # use the exactly same adj and obj for the phrase
            mc_rel = random.choice(self.rel_choices[rel]['eq'])

            # Fill in
            mc_rel_phrase = self.rel_choices[mc_rel].mc_phrase
            mc_rel_phrase = mc_rel_phrase.replace('adj2',
                                                  adj2).replace('obj2', obj2)
            mc = mc_template.substitute(
                action=mc_action,
                adj1='',
                obj1=obj1_phrase,
                rel_phrase=mc_rel_phrase,
            )
            # reduce double empty space
            mc = mc.replace('  ', ' ')
            mc_all.append(mc)

        # Return mc and info
        info = {
            'num_amb_mc': 0,
            'num_amb_mc_possible': 100
        }  # make a big number, since attribute can be very ambiguous
        return mc_all, mc_types, info
