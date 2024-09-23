import argparse
import gc
import os
import sys

import jsonlines
import requests
import numpy as np
import json
import time
import torch
from pathlib import Path
from transformers import StoppingCriteriaList, StoppingCriteria
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

if __name__ == "__main__":
    #check_version()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./Phind-CodeLlama-34B-v2",
        help="Type of Codex Model to run",
    )
    parser.add_argument(
        "--source_file",
        type=str,
        default="./ds1000_data",
        help="Path to the downloaded DS-1000 data",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature of the Codex sampling distribtuion.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p cutoff of the Codex sampling distribtuion",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Number of maximum tokens for Model to generate",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of requests to issue at one time",
    )
    parser.add_argument(
        "--result_file_name",
        type=str,
        default='result-0611.json',
        help="Generated result file name",
    )
    args = parser.parse_args()

    filepath = args.source_file

    with open(filepath, "rb") as f:
        part_prompt_data = [item for item in jsonlines.Reader(f)][0]  
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    # avoid huggingface/tokenizers process dead lock
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.temperature == 0:
        args.top_p = 1.0
    if 'Llama-3' in args.model:

        llm = LLM(model=args.model, tensor_parallel_size=len(available_gpus), gpu_memory_utilization=0.95, trust_remote_code=True,
                  max_model_len=16384, max_seq_len_to_capture=16384,
                  swap_space=20, enforce_eager=True, disable_custom_all_reduce=True)
    else:
        llm = LLM(model=args.model, tensor_parallel_size=len(available_gpus), gpu_memory_utilization=0.95, trust_remote_code=True,
                  max_model_len=16384, max_seq_len_to_capture=16384)

    sampling_params = SamplingParams(n=args.num, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, stop=['</s>','<|end_of_text|>'], stop_token_ids=[2,128001],
                                     presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                                     top_k=-1, min_p=0.0, use_beam_search=False, length_penalty=1.0, early_stopping=False, include_stop_str_in_output=False,
                                     ignore_eos=False, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True)


    for prompt_path in tqdm(part_prompt_data.keys()):
        prompts = []
        prompt_name_dict = {}
        prompt_dict = part_prompt_data[prompt_path]  # {prompt_name:prompt}
        prompt_names = list(prompt_dict.keys())
        for prompt_name in prompt_names:
            prompt = prompt_dict[prompt_name]
            prompts.append(prompt)
            # prompt_name_dict[prompt] = prompt_name
        try:
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        # outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id

        #     for i in range(2):
        #         if len(outputs) >= len(prompt_names):
        #             break
        #         print(f'\n{prompt_path} generate {len(outputs)} prompt')
        #         outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

            res = {}
            for output in outputs:
                current_request_id = int(output.request_id) % len(prompt_names)
                prompt_name = prompt_names[current_request_id]
                res[prompt_name] = [o.text for o in output.outputs if len(o.text) > 0][:10] 
                if len(res[prompt_name]) == 0:
                    print(f'{prompt_path} {prompt_name} : {len(res[prompt_name])}')

            with open(os.path.join(prompt_path, args.result_file_name), 'w+') as f:
                json.dump(res, f)
                f.flush()
        except RuntimeError as e: 
            print('runtime error')
            print(e)
            print(sys.exc_info())

            del llm.llm_engine.model_executor
            del llm
            gc.collect()
            torch.cuda.empty_cache()

            if 'Llama-3' in args.model:

                llm = LLM(model=args.model, tensor_parallel_size=len(available_gpus), gpu_memory_utilization=0.95, trust_remote_code=True,
                          max_model_len=16384, max_seq_len_to_capture=16384,
                          swap_space=32, enforce_eager=True, disable_custom_all_reduce=True)
                # llm = LLM(model=args.model, tensor_parallel_size=len(available_gpus), gpu_memory_utilization=0.95, trust_remote_code=True,
                #           max_model_len=16384, max_seq_len_to_capture=16384,
                #           swap_space=16)
            else:
                llm = LLM(model=args.model, tensor_parallel_size=len(available_gpus), gpu_memory_utilization=0.95, trust_remote_code=True,
                          max_model_len=16384, max_seq_len_to_capture=16384)
        except Exception as e:
            print('error')
            print(e) 
            print(sys.exc_info())
            # del a vllm.executor.ray_gpu_executor.RayGPUExecutor object


