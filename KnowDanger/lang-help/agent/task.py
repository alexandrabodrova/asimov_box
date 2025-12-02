"""
Task agent for generating requests.

"""

import random
from string import Template
from omegaconf import OmegaConf

from agent.ambiguity import AttributeAmbiguity, SpatialAmbiguity, NumericAmbiguity, SyntacticAmbiguity, ATTRIBUTE_AMBIGUITY, SPATIAL_AMBIGUITY, NUMERIC_AMBIGUITY, SYNTACTIC_AMBIGUITY, ambiguity_names


class Task:

    def __init__(self, cfg):
        self.request_template_choices = cfg.request_template_choices
        self.ambiguity_ratio = cfg.ambiguity_ratio

        # Initialize factories for each ambiguity type
        self.attribute_factory = AttributeAmbiguity(
            cfg,
            cfg.numeric_choices,
            cfg.adj_choices,
            cfg.obj_choices,
            cfg.rel_choices,
            cfg.action_choices,
            cfg.mc_template_choices,
        )
        self.spatial_factory = SpatialAmbiguity(
            cfg,
            cfg.numeric_choices,
            cfg.adj_choices,
            cfg.obj_choices,
            cfg.rel_choices,
            cfg.action_choices,
            cfg.mc_template_choices,
        )
        self.numeric_factory = NumericAmbiguity(
            cfg,
            cfg.numeric_choices,
            cfg.adj_choices,
            cfg.obj_choices,
            cfg.rel_choices,
            cfg.action_choices,
            cfg.mc_template_choices,
        )
        self.syntactic_factory = SyntacticAmbiguity(
            cfg,
            cfg.numeric_choices,
            cfg.adj_choices,
            cfg.obj_choices,
            cfg.rel_choices,
            cfg.action_choices,
            cfg.mc_template_choices,
        )

    def sample_request(self):
        ambiguity_type = random.choices(
            [
                ATTRIBUTE_AMBIGUITY,
                SPATIAL_AMBIGUITY,
                NUMERIC_AMBIGUITY,
                SYNTACTIC_AMBIGUITY,
            ],
            weights=self.ambiguity_ratio,
            k=1,
        )[0]
        if ambiguity_type == ATTRIBUTE_AMBIGUITY:
            factory = self.attribute_factory
        elif ambiguity_type == SPATIAL_AMBIGUITY:
            factory = self.spatial_factory
        elif ambiguity_type == NUMERIC_AMBIGUITY:
            factory = self.numeric_factory
        elif ambiguity_type == SYNTACTIC_AMBIGUITY:
            factory = self.syntactic_factory
        else:
            raise 'Unknown ambiguity type!'

        # Sample request
        request_template = Template(
            random.choice(self.request_template_choices)
        )

        # Sample request setting (objects, adjectives, and relations) based on the ambiguity type
        action, obj1, adj1, rel, obj2, adj2, _ = factory.sample_request_setting(
        )
        # TODO: save ambiguity flag for fine-tuning

        # Sample a ground truth request - ignore action right now
        obj1_true, adj1_true, rel_true, obj2_true, adj2_true = factory.sample_ground_truth_words(
            obj1,
            adj1,
            rel,
            obj2,
            adj2,
        )

        # Generate request according to the ambiguity type
        request = factory.generate_request(
            request_template,
            action,
            obj1,
            adj1,
            rel,
            obj2,
            adj2,
        )
        request = request.replace(' ,', ',')

        # Generate ground truth request with the same action
        request_unambiguous = factory.generate_request(
            request_template,
            action,
            obj1_true,
            adj1_true,
            rel_true,
            obj2_true,
            adj2_true,
        )
        request_unambiguous = request_unambiguous.replace(' ,', ',')

        # Generate scene description
        object_set = factory.sample_object_set()

        # Extra info
        info = OmegaConf.create()
        info.ambiguity_type = ambiguity_type
        info.ambiguity_name = ambiguity_names[ambiguity_type]
        if 'None' in request_unambiguous:
            request_unambiguous = "Invalid request"
        info.request_unambiguous = request_unambiguous
        info.object_set = object_set
        info.sample_words = {
            'action': action,
            'obj1': obj1,
            'adj1': adj1,
            'rel': rel,
            'obj2': obj2,
            'adj2': adj2
        }
        info.true_words = {
            'action': action,
            'obj1': obj1_true,
            'adj1': adj1_true,
            'rel': rel_true,
            'obj2': obj2_true,
            'adj2': adj2_true
        }
        return request, info

    def generate_mc(self, obj1, adj1, rel, obj2, adj2, ambiguity_type):
        if ambiguity_type == ATTRIBUTE_AMBIGUITY:
            factory = self.attribute_factory
        elif ambiguity_type == SPATIAL_AMBIGUITY:
            factory = self.spatial_factory
        elif ambiguity_type == NUMERIC_AMBIGUITY:
            factory = self.numeric_factory
        elif ambiguity_type == SYNTACTIC_AMBIGUITY:
            factory = self.syntactic_factory
        else:
            raise 'Unknown ambiguity type!'
        return factory.generate_mc(obj1, adj1, rel, obj2, adj2)
