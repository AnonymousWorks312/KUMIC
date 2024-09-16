import argparse
import json
import re

import numpy as np
import openai
from jsonlines import jsonlines
from openai import OpenAI
import os
from tqdm import tqdm
import jieba.posseg as pseg
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import subprocess

prompt_template_codellama = '### System Prompt\n{}\n\n### User Message:\n{}\n\n### Assistant:'
prompt_template_llama3 = ("<|begin_of_text|>"
                          "<|start_header_id|>system<|end_header_id|>"
                          "{}"
                          "<|eot_id|>"
                          "<|start_header_id|>user<|end_header_id|>"
                          "{}"
                          "<|eot_id|>"
                          "<|start_header_id|>assistant<|end_header_id|>")

prompt_templates = {'codellama': prompt_template_codellama,
                    'llama3': prompt_template_llama3}

prompt_types = {'all': ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
                        'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
                        'cot_0shot_comment2code', 'cot_3shot_comment2code', 'cot_5shot_comment2code'],
                'all_no_zero': ['baseline_3shot_comment2code', 'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
                                'cot_3shot_comment2code', 'cot_5shot_comment2code'],
                'cot': ['cot_0shot_comment2code', 'cot_3shot_comment2code', 'cot_5shot_comment2code'],
                'cot_no_zero': ['cot_3shot_comment2code', 'cot_5shot_comment2code'],
                'no_cot': ['baseline_3shot_comment2code'],
                'cg': ['cot_3shot_cg'],
                'fix': ['baseline_3shot_comment2code','cot_3shot_comment2code'],
                'one': ['cot_3shot_comment2code']
                }


def run_codellama_api(prompt_path):
    # # Configuration for custom OpenAI endpoint
    # openai.api_key = "EMPTY"
    # openai.api_base = "http://172.28.102.2:8101/v1"  # Ensure this is just the base URL
    #
    # # Chat completions are a special endpoint that may differ
    # response = openai.ChatCompletion.create(
    #   model="Phind-CodeLlama-34B-v2",
    #   messages=[{"role": "user", "content": prompt}],
    # )
    with open(prompt_path, "r") as f:
        prompt = f.read()

    client = OpenAI(api_key="EMPTY", base_url="http://172.28.102.2:8101/v1")
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="Phind-CodeLlama-34B-v2",
        messages=messages,
        temperature=0.5,
        top_p=0.8,
        max_tokens=4096
    )
    return response.choices[0].message.content


def get_comment_from_file(prompt_types, path="retrieve_sim_token", result_file_name='0619'):
    print(path)
    prompt_types_len = len(prompt_types)
    # prompt_types = ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
    #                 'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
    #                 'cot_0shot_comment2code', 'cot_3shot_code2comment',
    #                 'cot_3shot_comment2code', 'cot_5shot_comment2code']
    # prompt_types = ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
    #                 'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
    #                 'cot_0shot_comment2code',
    #                 'cot_3shot_comment2code', 'cot_5shot_comment2code']

    intents = ['what', 'why', 'done', 'property', 'usage']
    res = {intent: {} for intent in intents}
    reformat_res = {intent: {} for intent in intents}

    for intent in intents:  # 遍历文件夹
        print(f"in [{intent}]")
        ids_path = os.path.join(path, f'{intent}')
        if os.path.isdir(ids_path):
            idss = os.listdir(ids_path)
            for ids in tqdm(idss):
                if not os.path.exists(os.path.join(ids_path, ids, f'result-{result_file_name}.json')):
                    print(f'Not exist [{os.path.join(ids_path, ids)}]')
                    continue
                res[intent][ids] = {}
                reformat_res[intent][ids] = {}
                with open(os.path.join(ids_path, ids, f'result-{result_file_name}.json'), 'r') as f:
                    prompttype_comments = json.loads(f.read())  # {prompt_type:comments}
                    # pre_num_per_prompt = [len(prompttype_comments[key]) for key in prompttype_comments.keys()]

                # 少预测了prompt或者没有预测出结果
                if len(prompttype_comments.keys()) < prompt_types_len:  # or min(pre_num_per_prompt) == 0:
                    print(f"fail=={os.path.join(ids_path, ids)}")
                    # print(prompttype_comments.keys())
                    # print(os.path.join(ids_path, ids))
                    continue

                for prompt_type in prompttype_comments.keys():
                    comments = prompttype_comments[prompt_type]
                    try:
                        if len(comments) == 0:
                            res[intent][ids][prompt_type] = ['None']
                            reformat_res[intent][ids][prompt_type] = ['None']
                        else:
                            res[intent][ids][prompt_type] = []
                            for comment in comments:
                                if comment != '' and len((r_comment := get_comment_by_re(comment)).split()) < 100:
                                    res[intent][ids][prompt_type].append(r_comment)

                            if len(res[intent][ids][prompt_type]) == 0:
                                res[intent][ids][prompt_type] = ['None']
                            reformat_res[intent][ids][prompt_type] = [reformat_comment(comment.lower()) for comment in res[intent][ids][prompt_type] if comment.strip() != '']
                        if len(reformat_res[intent][ids][prompt_type]) == 0:
                            reformat_res[intent][ids][prompt_type] = ['None']
                    except Exception as e:
                        print(f"{e}")
                        print(f'{os.path.join(ids_path, ids)} : {prompt_type}')

                # if len(list(res[intent][ids].keys())) < len(prompt_types):
                #    print(len(list(res[intent][ids].keys())))
                #    exit(0)

    # with open(f"{path}/comment-{result_file_name}.pred", 'w+') as f:
    #     json.dump(res, f)
    with open(f"{path}/reformat_comment-{result_file_name}.pred", 'w+') as f:
        json.dump(reformat_res, f)


def get_comment_by_re(llm_result):
    llm_result = llm_result.replace('\n', ' ')
    start_tokens = ['The comment.*?:.*? method (which|that|for|to)',
                    'The comment.*?: Test method.*? (which|that)',
                    'The comment.*?: This method is (expected|provided|designed|used) to',
                    'The comment.*?: This.*? method is responsible for',
                    'The comment.*?: This.*? method',
                    'The comment.*?: The.*? method',
                    'The comment.*?: This test code',
                    'The comment.*?: The test code',
                    'The comment.*?: Test code',
                    'The comment.*?: [Tt]he (code|function|method)',
                    'The comment.*?: [Tt]his (code|function|method)',
                    'The comment.*?: [Aa]ttempt(s*) to',
                    'The comment.*?: [mM]ethod',
                    'The comment.*?:',
                    'The implementation of the test code',
                    'This test code defines a method .*?(which|that|for|to)',
                    'This test code is a method in.*? (which|that|for|to)',
                    'This test code is a method named .*? (which|that|for|to)',
                    'The test code defines a method .*?(which|that|for|to)',
                    'This test code.*? is (expected|provided|designed|used) to',
                    'The test code.*? is (expected|provided|designed|used) to',
                    'The test code is a.*? method (that)*',
                    'The test code method `.*?`',
                    'The test code is an implementation of .*?(which|that)',
                    'This Java code is an implementation of.*? method, which',  # This Java code is an implementation of the actionPerformed method, which
                    'This code defines .*? method (which|that|for|to)',
                    'This method.*? is.*?function (which|that|for|to)',  # This method, `addReplicationFactor`, is a utility function that
                    'This method.*? is (expected|provided|designed|used) to',
                    'This method attempt(s*) to',
                    'This.*? method,.*?,',  # This method, `convertSWFToHTML`, takes an input stream as an argumen
                    'The.*? method.*? is (expected|provided|designed|used) to',
                    'The method.*?is .*?method (which|that|for|to)',
                    'The.*? method `.*?`',
                    'The.*? method',
                    'The test code',
                    'This test code',
                    'This test method',
                    '[Tt]his (code|function|method)',
                    '[Tt]he (code|function|method)'
                    ]
    comment = ''
    for start_token in start_tokens:
        tmp = re.search(rf"{start_token} .*", llm_result)
        if tmp is not None:
            # print(tmp.group())
            raw_comment = tmp.group()
            tmp = re.search(rf"{start_token} .*?\.", raw_comment)
            raw_comment = tmp.group() if tmp is not None else raw_comment

            comment = re.sub(start_token, '', raw_comment)
            break
    if not comment.strip() and '.' in llm_result:
        comment = re.search(r'.*?(\.)', llm_result).group()  # 选择第一句话
    elif not comment.strip():
        comment = llm_result

    if comment[-1] == '.':
        comment = comment[:-1]

    return comment


def reformat_comment(comment):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(comment)
    tags = nltk.pos_tag(words)
    # 将句子中所有动词的单数换为复数
    for idx, word in enumerate(tags):
        if word[1] == 'VBZ':
            words[idx] = lemmatizer.lemmatize(words[idx], pos='v')
    if tags[0][1] != 'VBD' and tags[0][1] != 'VBN':
        words[0] = lemmatizer.lemmatize(words[0], pos='v')
    return " ".join(words)


def find_unpred_prompt_file(prompt_types, path="retrieve_sim_token", result_file_name='result-0623.json', llm_type='llama3', split_num=1):
    print(path)
    # prompt_types = ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
    #                 'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
    #                 'cot_0shot_comment2code', 'cot_3shot_code2comment',
    #                 'cot_3shot_comment2code', 'cot_5shot_comment2code']
    # prompt_types = ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
    #                 'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
    #                 'cot_0shot_comment2code',
    #                 'cot_3shot_comment2code', 'cot_5shot_comment2code']
    # prompt_types = ['cot_0shot_comment2code', 'cot_3shot_comment2code', 'cot_5shot_comment2code']
    intents = ['what', 'why', 'done', 'property', 'usage']
    res = {intent: {} for intent in intents}
    reformat_res = {intent: {} for intent in intents}
    rest_prompt_data = {}  # 得到文件夹下的所有文件名称

    def load_prompt_data(prompt_files_path):
        rest_prompt_data[prompt_files_path] = {}
        prompt_names = os.listdir(prompt_files_path)
        for prompt_name in prompt_names:
            for prompt_type in prompt_types:
                if prompt_type in prompt_name and 'txt' in prompt_name and not os.path.isdir(os.path.join(prompt_files_path, prompt_name)):
                    with open(os.path.join(prompt_files_path, prompt_name), "r") as f:
                        prompt = f.read()
                        rest_prompt_data[prompt_files_path][prompt_type] = prompt_templates[llm_type].format('You are an expert Java programmer.', prompt)

    for intent in intents:  # 遍历文件夹
        print(f"in [{intent}]")
        ids_path = os.path.join(path, f'{intent}')
        if os.path.isdir(ids_path):
            idss = os.listdir(ids_path)
            for ids in tqdm(idss):
                res[intent][ids] = {}
                reformat_res[intent][ids] = {}
                res_path = os.path.join(ids_path, ids, result_file_name)
                if os.path.exists(res_path):
                    # 不检查是否预测全
                    with open(res_path, 'r') as f:
                        prompttype_comments = json.loads(f.read())  # {prompt_type:comments}
                        pre_num_per_prompt = [len(prompttype_comments[key]) for key in prompttype_comments.keys()]

                    if min(pre_num_per_prompt) == 0:
                        prompt_files_path = os.path.join(ids_path, ids)
                        load_prompt_data(prompt_files_path)
                        print(prompt_files_path)
                    else:
                        # 少预测了prompt或者没有预测出结果
                        for prompt_type in prompt_types:
                            if prompt_type not in prompttype_comments.keys():
                                prompt_files_path = os.path.join(ids_path, ids)
                                load_prompt_data(prompt_files_path)
                                print(prompt_files_path)
                                break

                else:
                    prompt_files_path = os.path.join(ids_path, ids)
                    load_prompt_data(prompt_files_path)

    for idx, split_keys in enumerate(np.array_split(np.array(list(rest_prompt_data.keys())), split_num)):
        part_data = {key: rest_prompt_data[key] for key in split_keys}
        with open(os.path.join(path, f"part_prompt_data_{llm_type}-{idx}.json"), "w+") as f:
            json.dump(part_data, f)

    print(f'rest prompt num = {len(rest_prompt_data.keys())}')
    return len(rest_prompt_data.keys())

def run_vllm(path, gpus=[2, 3], llm_type='llama3', date='0623'):
    processes = []
    if llm_type == 'llama3':
        per_dp_rank_cards = 2
        model = "./Llama-3-70B-Instruct-Gradient-262k"
    else:
        per_dp_rank_cards = 1
        model = "./Phind-CodeLlama-34B-v2"
    dp_size = len(gpus) // per_dp_rank_cards
    # load_and_save_prompt(path, split_num=dp_size)
    cmds = []
    print(dp_size)
    for i in range(dp_size):
        cuda_device = ""
        for j in range(per_dp_rank_cards):
            cuda_device += str(gpus[i * per_dp_rank_cards + j]) + ","
        data_path = f'{path}/part_prompt_data_{llm_type}-{i}.json'
        # data_path = f'{path}/rest_prompt_data.json'
        cmd = (f"CUDA_VISIBLE_DEVICES={cuda_device[:-1]} python run_inference_vllm.py "
               f"--model {model} "
               f"--source_file {data_path} "
               f"--result_file_name result-{date}.json "
               f"--temperature 0.5 "
               f"--top_p 0.8 "
               f"--num 10 "
               f"--max_tokens 4096 > run_vllm_stdout-{i}.txt")

        cmds.append(cmd)

    for cmd in cmds:
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()

    for i, p in enumerate(processes):
        print(f'子进程{i}已经结束，退出状态码为：{p.returncode}')

    print("success")


if __name__ == '__main__':


    path = '../tlcodesum/retrieve_token'

    prompt_type4path = 'all'
    date = '0909'
    llm = 'codellama'


    rest_prompt_num = find_unpred_prompt_file(prompt_types=prompt_types[prompt_type4path],
                                                  path=path,
                                                  result_file_name=f'result-{date}.json',
                                                  llm_type=llm,
                                                  split_num=4)

    if rest_prompt_num != 0:
        run_vllm(path=path[idx],
                     llm_type=llm,
                     gpus=[0, 1, 2, 3],
                     date=date)

    get_comment_from_file(prompt_types[prompt_type4path],
                          path,
                          result_file_name=date)
