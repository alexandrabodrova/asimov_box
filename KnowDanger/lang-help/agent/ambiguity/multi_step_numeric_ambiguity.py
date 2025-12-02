"""Numeric type ambiguity, multi-step setting.

Assume there is only one correct answer for each step.

Example:
    request: put all blocks into the green bowl
    choices (first step):
        a) put the yellow block in the green bowl
        b) put the blue bowl in the green bowl
        c) put the green block in the blue bowl
        d) put the red block in the green bowl
    answer: a)

    choices (second step):
        a) put the yellow block in the green bowl
        b) put the green bowl in the green bowl
        c) put the green block in the blue bowl
        d) put the blue block in the green bowl
    answer: d)

    choices (third step):
        a) put the yellow block in the green bowl
        b) put the green block in the blue bowl
        c) put the green block in the green bowl
        d) put the blue block in the green bowl
    answer: c)

Use a new possible option: 'do nothing since the task is completed'. If the robot instead chooses to keep moving things, it is considered wrong.

We always use true attributes and objects here, such as yellow block or green bowl, in the multiple choices. TODO: relax this design choice.

"""
import random
import logging

from .numeric_ambiguity import NumericAmbiguity
from .multi_step_ambiguity import MultiStepAmbiguity


class MultiStepNumericAmbiguity(NumericAmbiguity):
    """
    Args:
        cfg (dict): configuration
        num_choices (dict): number of objects
        adj_choices (dict): adjectives and related words
        obj_choices (dict): objects and related words
        rel_choices (dict): relations and related words
        action_choices (dict): actions and related words
        mc_template_choices (list): templates of multiple choice
    """

    def __init__(
        self,
        cfg,
        num_choices,
        adj_choices,
        obj_choices,
        rel_choices,
        action_choices,
        mc_template_choices,
    ):
        super().__init__(
            cfg, num_choices, adj_choices, obj_choices, rel_choices,
            action_choices, mc_template_choices
        )
        self.multi_step = MultiStepAmbiguity(cfg)  # composition for now

        # always use correct num for now
        assert self.correct_ratio == 1

    def sample_request_setting(self):
        """Make sure the question is eq type, i.e., number of objects is not ambiguous."""
        while 1:
            action, obj1, num, rel, obj2, adj2 = super(
            ).sample_request_setting()
            if len(self.num_choices[num]['eq']) > 0:
                break
        flag_ambiguous = False  # TODO: placeholder
        return action, obj1, num, rel, obj2, adj2, flag_ambiguous

    def generate_mc_steps(self, obj1, num, rel, obj2, adj2, exclude_sem=False):

        # Check if sem target reached
        # if exclude_sem:
        #     logging.info('Sem target reached and no amb/eq mc exists!\n')
        #     return None

        # Substitute obj1, obj2, adj2 with eq
        obj1 = random.choice(self.obj_choices[obj1]['eq'])
        obj2 = random.choice(self.obj_choices[obj2]['eq'])
        adj2 = random.choice(self.adj_choices[adj2]['eq'])

        # Check if numeric relation exists - assume no amb right now
        mc_num = random.choice(self.num_choices[num]['eq'])

        # Generate the correct option for all steps - might be fewer than num_step, as we will also use do nothing
        true_attribute_steps = random.sample(self.true_adj_choices, k=mc_num)

        # # Whether to include the correct one in the mc
        # flag_use_correct = random.random() < self.correct_ratio

        # Loop through steps
        mc_all_steps = []
        mc_types_steps = []
        for step in range(self.multi_step.num_step):

            # Generate multiple choices of the current step
            mc_all = []
            mc_types = []

            # true attribute for this step
            flag_step_use_true = False
            if step < len(true_attribute_steps):
                flag_step_use_true = True
                true_attribute = true_attribute_steps[step]

            # each option - always put the correct one in the first option (fine since we shuffle in post-processing), and always put do nothing in the last option
            for option_ind in range(self.num_mc_sample):

                # Sample the template
                mc_template = self.sample_mc_templete()

                # do nothing as last option
                if option_ind == self.num_mc_sample - 1:
                    mc = 'do nothing (task already completed)'

                    # if there is still true attribute left, then it is wrong to do nothing
                    if flag_step_use_true:
                        mc_types.append('sem')
                    else:
                        mc_types.append('eq')
                else:
                    # use the correct one if exists
                    if option_ind == 0 and flag_step_use_true:
                        attribute_option = true_attribute
                        obj1_option = obj1
                        mc_types.append('eq')

                        # use the correct relation
                        mc_rel_option = random.choice(
                            self.rel_choices[rel]['eq']
                        )

                    # sample random ones - random attribute and random object,
                    else:
                        while 1:
                            attribute_option = random.choice(
                                list(self.adj_choices.keys())
                            )
                            obj1_option = random.choice(
                                list(self.obj_choices.keys())
                            )

                            # make sure the object is sem to the true one
                            if obj1_option in self.obj_choices[obj1]['sem']:

                                # also check if the current step uses true, do not generate sem option from true_attributes
                                if not (
                                    flag_step_use_true and attribute_option
                                    in self.true_adj_choices and (
                                        obj1_option
                                        in self.obj_choices[obj1]['eq'] or obj1
                                        in self.obj_choices[obj1_option]['eq']
                                    )
                                ):
                                    break

                        # use a random option for relation
                        mc_rel_option = random.choice(
                            self.rel_choices[rel]['eq']
                            + self.rel_choices[rel]['sem']
                        )

                        # there are still cases like "put the yellow dish in the green block" while the true one is "put the blue bowl in the green block". It would be better to mark it as amb, but we mark it as sem for now.
                        mc_types.append('sem')

                        # We are also avoiding the case where the last step is "put the yellow block next to the green block", and use the same again here. It could be wrong if there are three steps and three blocks need to be moved, or could be okay if fewer than three blocks need to be moved.

                        # There could be many other corner cases that can be addressed to improve the quality of the dataset.

                    # Substitute in adj2 and obj2
                    mc_rel_option_phrase = self.rel_choices[mc_rel_option
                                                           ].mc_phrase
                    mc_rel_option_phrase = mc_rel_option_phrase.replace(
                        'adj2', adj2
                    ).replace('obj2', obj2)

                    # get mc
                    mc_action_option = random.choice(self.action_choices)
                    mc = mc_template.substitute(
                        action=mc_action_option,
                        adj1=attribute_option,
                        obj1=obj1_option,
                        rel_phrase=mc_rel_option_phrase,
                    )

                # save
                mc_all.append(mc)

            # save
            mc_all_steps.append(mc_all)
            mc_types_steps.append(mc_types)

        # Return mc and info
        info_steps = [{
            'num_amb_mc': 0
        } for _ in range(self.multi_step.num_step)]
        return mc_all_steps, mc_types_steps, info_steps
