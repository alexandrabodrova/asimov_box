"""
Multi-step setting.

"""


class MultiStepAmbiguity():
    """
    Args:
        cfg (dict): config
    """

    def __init__(self, cfg):

        # cfg
        self.num_step = cfg.num_step
