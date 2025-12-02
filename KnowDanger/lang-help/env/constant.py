import numpy as np


PICK_TARGETS = {
    'blue block': None,
    'red block': None,
    'green block': None,
    'orange block': None,
    'yellow block': None,
    'purple block': None,
    'pink block': None,
    'cyan block': None,
    'brown block': None,
    'gray block': None,
}

COLORS = {
    'blue': (78 / 255, 121 / 255, 167 / 255, 255 / 255),
    'red': (255 / 255, 87 / 255, 89 / 255, 255 / 255),
    'green': (89 / 255, 169 / 255, 79 / 255, 255 / 255),
    'orange': (242 / 255, 142 / 255, 43 / 255, 255 / 255),
    'yellow': (237 / 255, 201 / 255, 72 / 255, 255 / 255),
    'purple': (176 / 255, 122 / 255, 161 / 255, 255 / 255),
    'pink': (255 / 255, 157 / 255, 167 / 255, 255 / 255),
    'cyan': (118 / 255, 183 / 255, 178 / 255, 255 / 255),
    'brown': (156 / 255, 117 / 255, 95 / 255, 255 / 255),
    'gray': (186 / 255, 176 / 255, 172 / 255, 255 / 255),
}

PLACE_TARGETS = {
    'blue block': None,
    'red block': None,
    'green block': None,
    'orange block': None,
    'yellow block': None,
    'purple block': None,
    'pink block': None,
    'cyan block': None,
    'brown block': None,
    'gray block': None,
    'blue bowl': None,
    'red bowl': None,
    'green bowl': None,
    'orange bowl': None,
    'yellow bowl': None,
    'purple bowl': None,
    'pink bowl': None,
    'cyan bowl': None,
    'brown bowl': None,
    'gray bowl': None,
    'top left corner': (-0.3 + 0.05, -0.2 - 0.05, 0),
    'top side': (0, -0.2 - 0.05, 0),
    'top right corner': (0.3 - 0.05, -0.2 - 0.05, 0),
    'left side': (-0.3 + 0.05, -0.5, 0),
    'middle': (0, -0.5, 0),
    'right side': (0.3 - 0.05, -0.5, 0),
    'bottom left corner': (-0.3 + 0.05, -0.8 + 0.05, 0),
    'bottom side': (0, -0.8 + 0.05, 0),
    'bottom right corner': (0.3 - 0.05, -0.8 + 0.05, 0),
}

PIXEL_SIZE = 0.00267857
WORKSPACE_HALF_DIM = 0.3
BOUNDS = np.float32([[-WORKSPACE_HALF_DIM, WORKSPACE_HALF_DIM],
                     [-0.5 - WORKSPACE_HALF_DIM, -0.5 + WORKSPACE_HALF_DIM],
                     [0, 0.15]])  # (X, Y, Z)
