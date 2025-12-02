import argparse
from omegaconf import OmegaConf
import pickle
from tqdm import tqdm
import openai

def LM(
    prompt,
    lm_model="text-davinci-002",
    max_tokens=128,
    temperature=0,
    stop=None,
    logprobs=None,
    top_p=1,
    logit_bias={},
):
    response = openai.Completion.create(
        engine=lm_model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        logprobs=logprobs,
        top_p=top_p,
        logit_bias=logit_bias,
    )
    return response, response["choices"][0]["text"].strip()


def main(cfg):

    # Base context from the colab - there is no spatial information of the objects.
    gpt3_prompt = """
    objects = ["cyan block", "yellow block", "brown block", "green bowl"]
    # move all the blocks to the top left corner.
    robot.pick_and_place("brown block", "top left corner")
    robot.pick_and_place("cyan block", "top left corner")
    robot.pick_and_place("yellow block", "top left corner")
    # put the yellow one the green thing.
    robot.pick_and_place("yellow block", "green bowl")
    # undo that.
    robot.pick_and_place("yellow block", "top left corner")
    objects = ["pink block", "gray block", "orange block"]
    # move the pinkish colored block on the bottom side.
    robot.pick_and_place("pink block", "bottom side")
    objects = ["orange block", "purple bowl", "cyan block", "brown bowl", "pink block"]
    # stack the blocks.
    robot.pick_and_place("pink block", "orange block")
    robot.pick_and_place("cyan block", "pink block")
    # unstack that.
    robot.pick_and_place("cyan block", "bottom left")
    robot.pick_and_place("pink block", "left side")
    objects = ["red block", "brown block", "purple bowl", "gray bowl", "brown bowl", "pink block", "purple block"]
    # group the brown objects together.
    robot.pick_and_place("brown block", "brown bowl")
    objects = ["orange bowl", "red block", "orange block", "red bowl", "purple bowl", "purple block"]
    # sort all the blocks into their matching color bowls.
    robot.pick_and_place("orange block", "orange bowl")
    robot.pick_and_place("red block", "red bowl")
    robot.pick_and_place("purple block", "purple bowl")
    objects = ["yellow bowl", "yellow block", "green bowl", "blue block", "green block", "blue bowl"]
    # put the yellow cube inside the yellow bowl.
    robot.pick_and_place("yellow block", "yellow bowl")
    objects = ["yellow bowl", "yellow block", "green bowl", "blue block", "green block", "blue bowl"]
    """

    # Context
    context = '# put the yellow cube in the blue bowl\n'
    context += '# Can you generate a few possible choices of actions in plain words?\n'
    context += '#'

    # Append
    full_prompt = gpt3_prompt + context

    # Prompt GPT-3
    response_full, response = LM(
        full_prompt,
        temperature=cfg.gpt_temperature,
        stop=['#', 'objects ='],
        # max_tokens=1,
        # logprobs=5,
        # top_p=1,
        # logit_bias={   # banning some tokens
        #     14202:-100,  # None
        #     6045:-100,   #  None (with space at front)
        #     198:-100,   # \n
        #     383:-100,   #  The (with space at front)
        #     464:-100,   # The
        #     27271:-100, # Place
        #     220:-100,   # (empty space)
        #     11588:-100, # Put
        # },
    )
    print(response_full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    if args.cfg_file == '':
        print('Using pre-defined parameters!')
        cfg = OmegaConf.create()
        cfg.gpt_temperature = 0.0
    else:
        cfg = OmegaConf.load(args.cfg_file)
    main(cfg)
