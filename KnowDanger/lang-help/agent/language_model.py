""" Functions for interacting with LMs.

TODO: 
1. API for Google models like PaLM.

"""
import openai
import signal


class timeout:

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def convert_messages_to_prompt(messages):
    """Helper function that converts message format used in chat mode to plain prompt string."""
    prompt = ''
    for ind, m in enumerate(messages):
        # if m['role'] == 'user':
        #     prompt += 'Human: '
        # elif m['role'] == 'assistant':
        #     prompt += 'Robot: '
        prompt += m['content']
        if ind < len(messages) - 1:
            prompt += '\n\n'
    return prompt


class LanguageModel:

    def __init__(self, api_key):
        self._api_key = api_key
        openai.api_key = api_key

    def prompt_gpt_complete(
        self,
        prompt,
        lm_model="text-davinci-003",
        max_tokens=128,
        temperature=0,
        logprobs=None,
        top_p=1,
        logit_bias={},
        stop_seq=None,
        timeout_seconds=3,
        api_key=None,
    ):
        """
        Prompt GPT completion mode.

        text-davinci-003 often hangs, so we need to retry on timeout.
        
        Sample response from the documentation:
        {
            "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            "object": "text_completion",
            "created": 1589478378,
            "model": "text-davinci-003",
            "choices": [
                {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": null,
                "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12
            }
        }
        """
        if logit_bias is not None:
            logit_bias = dict(logit_bias)
        else:
            logit_bias = {}
        if api_key is not None:
            openai.api_key = api_key
        while 1:
            try:
                with timeout(seconds=timeout_seconds):
                    response = openai.Completion.create(
                        model=lm_model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        logprobs=logprobs,
                        top_p=top_p,
                        logit_bias=logit_bias,
                        stop=list(stop_seq) if stop_seq is not None else None,
                    )
                break
            except:
                print('Timeout, retrying...')
                pass
        return response, response["choices"][0]["text"].strip()

    def prompt_gpt_chat(
        self,
        messages,
        lm_model='gpt-3.5-turbo',
        max_tokens=128,
        temperature=0,
        stop=None,
        timeout_seconds=3,
        api_key=None,
    ):
        """
        Prompt GPT chat mode.

        Here the prompt/message is in a different format from the completion mode. Example from the documentation:
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
        
        Sample response from the documentation:
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        """
        if api_key is not None:
            openai.api_key = api_key
        while 1:
            try:
                with timeout(seconds=timeout_seconds):
                    response = openai.ChatCompletion.create(
                        messages=messages,
                        model=lm_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                    )
                break
            except:
                print('Timeout, retrying...')
                pass
        return response, response["choices"][0]["message"]["content"].strip()
