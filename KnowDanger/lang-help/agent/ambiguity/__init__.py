from .base_ambiguity import BaseAmbiguity
from .attribute_ambiguity import AttributeAmbiguity
from .spatial_ambiguity import SpatialAmbiguity
from .numeric_ambiguity import NumericAmbiguity
from .syntactic_ambiguity import SyntacticAmbiguity
from .multi_step_numeric_ambiguity import MultiStepNumericAmbiguity
from .multi_step_ambiguity import MultiStepAmbiguity


ATTRIBUTE_AMBIGUITY = 0
SPATIAL_AMBIGUITY = 1
NUMERIC_AMBIGUITY = 2
SYNTACTIC_AMBIGUITY = 3

ambiguity_names = {
    ATTRIBUTE_AMBIGUITY: 'attribute',
    SPATIAL_AMBIGUITY: 'spatial',
    NUMERIC_AMBIGUITY: 'numeric',
    SYNTACTIC_AMBIGUITY: 'syntactic',
}
