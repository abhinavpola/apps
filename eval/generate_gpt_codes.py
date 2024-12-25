"""
Run a tranined model to generate Python code.
"""

import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch

from datasets import load_dataset
from reindent import run as run_reindent
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm

from openai import OpenAI
import openai


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                print(f"Exponential backoff for {delay} seconds")
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def generate_openai_response(client, chat_args):
    response = client.chat.completions.create(**chat_args)
    return response.choices[0].message.content

def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False,
        },
    )

    return ret.getvalue()


def generate_prompt(
    args, test_case, prompt, solutions, tokenizer=None, starter_code=None
):
    _input = "\nQUESTION:\n"
    data = prompt
    _input += data
    if starter_code != None:
        data = starter_code
        data = "\n" + data  # + "\n"
        _input += data
    else:
        # _input += "\n\n"
        pass

    data = test_case
    if not data.get("fn_name"):
        _input += "\nOutput your answers to the STDOUT stream, such as by using print statements."  # \n"
    else:
        _input += "\nThe solution is to be provided as the function's return value."  # \n"
    _input += "\nOUTPUT THE CODE ONLY, NO OTHER TEXT."
    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking.

        # Read one example solution
        sols = solutions

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        if tokenizer:
            rand_sol = tokenizer.encode(rand_sol, verbose=False)
            tokens_taken = int(args.peek_frac * len(rand_sol))
            rand_sol = rand_sol[:tokens_taken]
            _input += tokenizer.decode(rand_sol)
        else:
            _input += rand_sol
    else:
        sample_sol = None

    return _input, sample_sol


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = load_dataset(
        "codeparrot/apps", split=f"{args.split}", trust_remote_code=True
    )

    gpt_codes = {}
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes.json")
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    # Only do the problems that are specified.
    if args.index:
        problems = load_dataset(
            "codeparrot/apps",
            split=f"{args.split}[{args.index}]",
            trust_remote_code=True,
        )
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = load_dataset(
            "codeparrot/apps",
            split=f"{args.split}[{start}:{end}]",
            trust_remote_code=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = None
    tokenizer = None
    if not args.model:
        if args.load:
            # Set up model
            # Tokenizer
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)
            print("Loading model...")
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
            model.to(device)
            print(f"Loaded {args.load}.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.arch)
            model = AutoModelForCausalLM.from_pretrained(
                args.arch, device_map="auto"
            ).eval()
    else:
        client = OpenAI()

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        problem["solutions"] = json.loads(problem["solutions"])
        problem["input_output"] = json.loads(problem["input_output"])
        test_case = problem["input_output"]
        prompt = problem["question"]
        starter_code = problem["starter_code"]
        solutions = problem["solutions"]
        if not starter_code:
            starter_code = None

        # Read the question in
        prompt_text, sample_sol = generate_prompt(
            args, test_case, prompt, solutions, tokenizer, starter_code
        )
        if args.debug:
            print(f"PROMPT_TEXT: \n{prompt_text}")

        # Feed this into the model.
        start = time.time()
        try:
            if args.model:
                chat_args = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": 1024,
                }
                response = generate_openai_response(client, chat_args)
                output_str = response
            else:
                with torch.no_grad():
                    input_ids = (
                        torch.LongTensor(tokenizer.encode(prompt_text, verbose=False))
                        .unsqueeze(0)
                        .to(device)
                    )
                    output_ids = model.generate(
                        input_ids,
                        num_beams=args.num_beams,
                        early_stopping=True,
                        max_length=1024 - len(input_ids),
                    )
                    output_str = tokenizer.decode(output_ids[0])
        except Exception as e:
            if (
                isinstance(e, UnboundLocalError)
                and str(e)
                == "local variable 'next_tokens' referenced before assignment"
            ):
                # See https://github.com/huggingface/transformers/issues/5118
                if args.debug:
                    print("Problem text was > 1024 tokens, so cannot do generation")
                    print(e)
            else:
                print("Unexpected exception in generating solution")
                print(e)
            # Default to empty string on errors
            output_str = ""
        end = time.time()

        if args.peeking == 1.0:
            output_str = sample_sol
        elif len(output_str):
            if not args.model:
                output_str = output_str.split("ANSWER:\n")[1].replace("<|endoftext|>", "")

        # Save the generated sol
        gpt_codes[index + args.start] = output_str

        if args.debug:
            print(f"Generation time: {end - start}")
            print(f"Generated output string:")
            print(output_str)
            print("------------------------------------------------------------")

    with open(codes_loc, "w") as f:
        json.dump(gpt_codes, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a tranined model to generate Python code."
    )
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument(
        "-t",
        "--test_loc",
        default="~/apps/data_split/test.json",
        type=str,
        help="path to the test folder.",
    )
    parser.add_argument(
        "-r", "--root", default="../", type=str, help="where the data is stored."
    )
    parser.add_argument("-l", "--load", default="", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("-e", "--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--split", type=str, default="test", help="What split to use.")
    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--model", type=str)

    args = parser.parse_args()

    main(args)
