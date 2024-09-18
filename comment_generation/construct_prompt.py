import random

import math
import os

import numpy as np
from jsonlines import jsonlines
from matplotlib import pyplot as plt
from tqdm import tqdm

exampler_what = "You are an expert Java programmer, please describe the functionality of the method:\n\"\"\"" + "Example Code1:\n"  # + example_code + "The comment is: Starts the background initialization"
exampler_why = "# You are an expert Java programmer, please explain the reason why the method is provided or the design rational of the method:\n\"\"\""  # + example_code + "The comment is: With this method the initializer becomes active and invokes the initialize() method in a background task"
exampler_use = "# You are an expert Java programmer, please describe the usage or the expected set-up of using the method:\n\"\"\""  # + example_code + "The comment is: After the construction of a BackgroundInitializer() object it start() method has to be called"
exampler_done = "# You are an expert Java programmer, please describe the implementation details of the method:\n\"\"\""  # + example_code + "The comment is: Get an external executor to create a background task. If there is not any, it creates a new one"
exampler_property = "# You are an expert Java programmer, please describe the asserts properties of the method including pre-conditions or post-conditions of the method:\n\"\"\""  # + example_code + "The comment is: Return the flag whether the initializer could be started successfully"
zero_what = "You are an expert Java programmer, please describe the functionality of the test code in a concise sentence:\n\"\"\""
zero_property = "# You are an expert Java programmer, please describe the asserts properties of the method including pre-conditions or post-conditions of the test code in a concise sentence:\n\"\"\""
zero_why = "# You are an expert Java programmer, please explain the reason why the method is provided or the design rational of the test code in a concise sentence:\n\"\"\""
zero_use = "# You are an expert Java programmer, please describe the usage or the expected set-up of using the test code in a concise sentence:\n\"\"\""
zero_done = "# You are an expert Java programmer, please describe the implementation details of the test code in a concise sentence:\n\"\"\""

head_prompt = "# You are an expert Java programmer, I will give you {} examples to learn from, and then you should help me "
mid_prompt = "# Please "
# head_prompt_cot = "# You are an expert Java programmer, I will give you {} examples to learn from, including the code, the important information extracted, and the corresponding code summary. Then you should help me extract the important information in the test code, and "
# mid_prompt_cot = "# Please imitate the above example, extract the important information in the test code, and then according to the above examples' comment "

head_prompt_cot = "# You are an expert Java programmer. There are NUM examples you should learn from, consist of the code, important statements in the code, and the corresponding code comment. You need to first analyse how to extract the important statements from the code and finally try to {} the code in a concise sentence according to the code and its important statements.\n# Note that important statements should be no more than ten lines and the comment provided should be as short as possible."
mid_prompt_cot = "# Please imitate the above example, extract important statements in the test code. Then analyse the code and  important statements to use one sentence to {} the test code in a high level of generality.\n# Note that important statements should be no more than ten lines and the comment provided should be as short as possible."

prompt_cot_info_augment = "# Step 1 - Important statements:"
prompt_cot_comment = "# Step 2 - The comment:"
m_what = "describe the functionality of"
m_property = "describe the asserts properties of the method including pre-conditions or post-conditions of"
m_why = "explain the reason why the method is provided or the design rational of"
m_use = "describe the usage or the expected set-up of using"
m_done = "describe the implementation details of"

prompt_lists = {'what': zero_what, 'why': zero_why, 'usage': zero_use, 'done': zero_done, 'property': zero_property}
intent_prompt_lists = {'what': m_what, 'why': m_why, 'usage': m_use, 'done': m_done, 'property': m_property}

head_prompt_lists = {'what': head_prompt + m_what, 'why': head_prompt + m_why, 'usage': head_prompt + m_use, 'done': head_prompt + m_done, 'property': head_prompt + m_property}
mid_prompt_lists = {'what': mid_prompt + m_what, 'why': mid_prompt + m_why, 'usage': mid_prompt + m_use, 'done': mid_prompt + m_done, 'property': mid_prompt + m_property}

head_prompt_cot_lists = {'what': head_prompt_cot.format(m_what), 'why': head_prompt_cot.format(m_why), 'usage': head_prompt_cot.format(m_use), 'done': head_prompt_cot.format(m_done), 'property': head_prompt_cot.format(m_property)}
mid_prompt_cot_lists = {'what': mid_prompt_cot.format(m_what), 'why': mid_prompt_cot.format(m_why), 'usage': mid_prompt_cot.format(m_use), 'done': mid_prompt_cot.format(m_done), 'property': mid_prompt_cot.format(m_property)}


def construct_prompts(selected_testcases_ids=[], dataset="tlcodesum", sim_rule="sim_semantic", intent="all", example_num=3,
                      is_quality=True, llm='cot', is_code_search_attention="comment2code", mode="train"):
    # is_code_search_attention=["comment2code","code2comment",None]
    def sort_normalize(attention_dict, max_line_num=10, min_line_num=3):
        '''
        :param attention_dict:根据正则化后的attention值对其排序
        '''

        def normal(arr):
            # 由于attention数值太少，使用sklearn的tranfer不大行
            res = []
            for x in arr:
                tmp = float(x - np.min(arr)) / (np.max(arr) - np.min(arr))
                res.append(tmp)
            return res

        normal_attention = {j: v for j, v in enumerate(normal(list(attention_dict.values())))}
        sorted_attention = sorted(normal_attention.items(), key=lambda x: x[1], reverse=True)
        line_id = [i[0] for i in sorted_attention]
        percentage = 0.4
        select_token_num = max(min(max_line_num, math.ceil(len(line_id) * percentage)), min_line_num)

        return sorted(line_id[:select_token_num])

    def generate_semantic_tokens4sentence(code_stat_attention_weight, ids2code_stat_format_lines, select_ids=[]):
        # semantic: [19495, 71743, 5245]
        # ast: [72116, 37755, 44859]
        def formate_semantic_tokens4sentance(sentance_list):
            res = []
            forbidden_word = ["try", "else {", "catch", "finally", "printStackTrace", "break", "continue"]
            for line_number, statement in sentance_list:
                flag = True
                for i in forbidden_word:
                    flag = flag and (i not in statement)

                if flag:
                    if ";" in statement[-1]:
                        res.append((line_number, statement.split(";")[0]))
                    elif "{" in statement[-1]:
                        res.append((line_number, statement[0:-1]))
                    else:
                        res.append((line_number, statement))
            return res

        ids2code_stat_attention_weight = {ids: attention_weight for ids, attention_weight in
                                          zip(ids2code_stat_format_lines.keys(), code_stat_attention_weight)}
        res = []
        for ids in select_ids:
            tmp = ""

            format_sentence = formate_semantic_tokens4sentance(
                [(int(line_number), ids2code_stat_format_lines[ids][int(line_number)])
                 for line_number in ids2code_stat_attention_weight[ids]])

            for line_id, statement in format_sentence:
                tmp += f"\n{line_id} {statement}"

            res.append(tmp)

        return res

    if example_num <= 0:
        with open(f"prompt_templates/prompt_{llm}_0shot", 'r', encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        with open(f"prompt_templates/prompt_{llm}", 'r', encoding="utf-8") as f:
            prompt_template = f.read()
    with open(f"dataset/{dataset}/train/intents/quality.train", 'r', encoding="utf-8") as f_quality:
        traincases_quality = [item for item in jsonlines.Reader(f_quality)][0]
    with open(f"dataset/{dataset}/test/intents/{intent}.test", "r+", encoding="utf-8") as f_test:
        testcases = {item["ids"]: item for item in jsonlines.Reader(f_test)}
    with open(f"dataset/{dataset}/train/intents/{intent}.train", "r+", encoding="utf-8") as f_train:
        traincases = {item["ids"]: item for item in jsonlines.Reader(f_train)}
    with open(f"dataset/{dataset}/test/intents/{sim_rule}/{sim_rule}_score.{intent}", "r+", encoding="utf8") as f_sim:
        sim_traincases_ids = {item["ids"]: item["sim_code"] for item in jsonlines.Reader(f_sim)}  # 只有一个json数据
    with open(f'./dataset/{dataset}/{mode}/code_split_format.{mode}', 'r') as f:  # 对code分行进行格式处理
        ids2code_stat_format_lines = {item["ids"]: item["code"] for item in jsonlines.Reader(f)}

    code_stat_attention_weight = None
    if is_code_search_attention == "comment2code":  # 读取模型attention信息
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight-cs.{mode}', 'r') as f:
            code_stat_attention_weight = [sort_normalize(item) for item in jsonlines.Reader(f)]
    elif is_code_search_attention == "code2comment":
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight-cs-20_code2comment.{mode}', 'r') as f:
            code_stat_attention_weight = [sort_normalize(item) for item in jsonlines.Reader(f)]
    else:
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight_-1.{mode}', 'r') as f:
            code_stat_attention_weight = [list(item.keys()) for item in jsonlines.Reader(f)]

    # 选择提取目标
    selected_testcases_ids = list(sim_traincases_ids.keys()) if len(selected_testcases_ids) == 0 else selected_testcases_ids
    print(len(selected_testcases_ids))

    for ids in tqdm(selected_testcases_ids):
        testcase = testcases[ids]
        testcase_code = testcase["raw_code"]
        testcase_comment = testcase["comment"]
        testcase_intent = testcase["label"]

        few_shot = ""

        sim_traincase_code_ids_score = sim_traincases_ids[ids]  # 选择相似example
        sim_score_standrad = 0.9 if sim_rule == 'sim_semantic' else 1

        example_score = np.array([not (is_quality and (sim_score <= sim_score_standrad and traincases_quality[sim_ids] < 0.5)) for (sim_ids, sim_score) in sim_traincase_code_ids_score])
        select = np.where(example_score != 0)[0]
        if len(select) < example_num:
            select = sorted(np.append(select, np.where(example_score != 0)[0][:example_num - len(select)]))

        for i, idx in enumerate(select[:example_num]):
            (sim_ids, sim_score) = sim_traincase_code_ids_score[idx]
            # 构造prompt
            # for i, (sim_ids, sim_score) in enumerate(sim_traincase_code_ids_score):
            sim_code = traincases[sim_ids]["raw_code"]
            sim_comment = traincases[sim_ids]["comment"]
            # sim_quality = traincases_quality[sim_ids]
            # if current >= example_num:  # 已收集足够example
            #     break
            #
            # if is_quality and (sim_score < 0.9 and sim_quality < 0.5):  # 质量未达标且相似度也未达标
            #     continue

            if llm == "baseline":
                few_shot += (f"\n#Example Code {i}:"
                             f"\n{sim_code}"
                             f"\n# The comment is: {sim_comment}")
            else:
                few_shot += (f"\n# Example Code {i}:"
                             f"\n{sim_code}"
                             f"\n{prompt_cot_info_augment}"
                             f"{generate_semantic_tokens4sentence(code_stat_attention_weight, ids2code_stat_format_lines, select_ids=[sim_ids])[0]}"
                             f"\n{prompt_cot_comment}"
                             f"\n{sim_comment}")
        if example_num <= 0:
            final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], intent_prompt_lists[testcase_intent], testcase_code, intent_prompt_lists[testcase_intent])
        else:
            if llm == "baseline":
                final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], few_shot, testcase_code)
            else:
                final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], intent_prompt_lists[testcase_intent], few_shot, testcase_code, intent_prompt_lists[testcase_intent])

        out_path = f"./{dataset}/retrieve_{sim_rule}/{testcase_intent}/{ids}"

        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/{ids}_{llm}_{example_num}shot_{str(is_code_search_attention)}.txt", "w+") as output:
            output.write(final_prompt)

def con_fix():
    example_num = 3
    dataset = "tlcodesum"
    intent = "all"
    with open(f"dataset/{dataset}/train/intents/{intent}.train", "r+", encoding="utf-8") as f_train:
        traincases = {item["ids"]: item for item in jsonlines.Reader(f_train)}

    sim_traincase_code_ids_score = random.sample(list(traincases.keys()), example_num)
    construct_prompts_with_fix_examples(sim_traincase_code_ids_score, llm='cot', example_num=example_num)
    construct_prompts_with_fix_examples(sim_traincase_code_ids_score, llm='baseline', example_num=example_num)

def construct_prompts_with_fix_examples(sim_traincase_code_ids_score, selected_testcases_ids=[], dataset="tlcodesum", sim_rule="sim_semantic", intent="all", example_num=3,
                                        llm='cot', is_code_search_attention="comment2code", mode="train"):
    def sort_normalize(attention_dict, max_line_num=10, min_line_num=3):
        '''
        :param attention_dict:根据正则化后的attention值对其排序
        '''

        def normal(arr):
            # 由于attention数值太少，使用sklearn的tranfer不大行
            res = []
            for x in arr:
                tmp = float(x - np.min(arr)) / (np.max(arr) - np.min(arr))
                res.append(tmp)
            return res

        normal_attention = {j: v for j, v in enumerate(normal(list(attention_dict.values())))}
        sorted_attention = sorted(normal_attention.items(), key=lambda x: x[1], reverse=True)
        line_id = [i[0] for i in sorted_attention]
        percentage = 0.4
        select_token_num = max(min(max_line_num, math.ceil(len(line_id) * percentage)), min_line_num)

        return sorted(line_id[:select_token_num])

    def generate_semantic_tokens4sentence(ids2code_stat_attention_weight, select_ids=[]):
        # semantic: [19495, 71743, 5245]
        # ast: [72116, 37755, 44859]
        def formate_semantic_tokens4sentance(sentance_list):
            res = []
            forbidden_word = ["try {", "else {", "catch ", "finally {", "printStackTrace", "break;", "continue;"]
            for line_number, statement in sentance_list:
                flag = True
                for i in forbidden_word:
                    flag = flag and (i not in statement)
                flag = flag and (statement != '}')

                if flag:
                    if statement.strip() == '':
                        continue
                    elif ";" in statement[-1]:
                        res.append((line_number, statement.split(";")[0]))
                    elif "{" in statement[-1]:
                        res.append((line_number, statement[0:-1]))
                    else:
                        res.append((line_number, statement))
            return res

        # ids2code_stat_attention_weight = {ids: attention_weight for ids, attention_weight in
        #                                   zip(ids2code_stat_format_lines.keys(), code_stat_attention_weight)}
        res = []
        for ids in select_ids:
            tmp = ""

            format_sentence = formate_semantic_tokens4sentance(
                [(int(line_number), ids2code_stat_format_lines[ids][int(line_number)])
                 for line_number in ids2code_stat_attention_weight[ids]])

            for line_id, statement in format_sentence:
                tmp += f"\n{line_id} {statement}"

            res.append(tmp)

        return res

    if example_num <= 0:
        with open(f"prompt_templates/prompt_{llm}_0shot", 'r', encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        with open(f"prompt_templates/prompt_{llm}", 'r', encoding="utf-8") as f:
            prompt_template = f.read()
    with open(f"dataset/{dataset}/test/intents/{intent}.test", "r+", encoding="utf-8") as f_test:
        testcases = {item["ids"]: item for item in jsonlines.Reader(f_test)}
    with open(f"dataset/{dataset}/train/intents/{intent}.train", "r+", encoding="utf-8") as f_train:
        traincases = {item["ids"]: item for item in jsonlines.Reader(f_train)}
    with open(f"dataset/{dataset}/test/intents/{sim_rule}/{sim_rule}_score.{intent}", "r+", encoding="utf8") as f_sim:
        sim_traincases_ids = {item["ids"]: item["sim_code"] for item in jsonlines.Reader(f_sim)}
    with open(f'./dataset/{dataset}/{mode}/code_split_format.{mode}', 'r') as f:  # 对code分行进行格式处理
        ids2code_stat_format_lines = {item["ids"]: item["code"] for item in jsonlines.Reader(f)}

    code_stat_attention_weight = None
    if is_code_search_attention == "comment2code":  # 读取模型attention信息
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight-cs.{mode}', 'r') as f:
            code_stat_attention_weight = [sort_normalize(item) for item in jsonlines.Reader(f)]
    elif is_code_search_attention == "code2comment":
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight-cs-20_code2comment.{mode}', 'r') as f:
            code_stat_attention_weight = [sort_normalize(item) for item in jsonlines.Reader(f)]
    else:
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight_-1.{mode}', 'r') as f:
            code_stat_attention_weight = [list(item.keys()) for item in jsonlines.Reader(f)]

    ids2code_stat_attention_weight = {ids: attention_weight for ids, attention_weight in
                                      zip(ids2code_stat_format_lines.keys(), code_stat_attention_weight)}
    # 选择提取目标
    selected_testcases_ids = list(sim_traincases_ids.keys()) if len(selected_testcases_ids) == 0 else selected_testcases_ids

    select = [i for i in range(example_num)]

    print(len(selected_testcases_ids))

    for ids in tqdm(selected_testcases_ids):

        try:
            testcase = testcases[ids]
            testcase_code = testcase["raw_code"]
            testcase_comment = testcase["comment"]
            testcase_intent = testcase["label"]

            few_shot = ""

            for i, idx in enumerate(select[:example_num]):
                sim_ids = sim_traincase_code_ids_score[idx]
                # 构造prompt
                sim_code = traincases[sim_ids]["raw_code"]
                sim_comment = traincases[sim_ids]["comment"]

                if llm == "baseline":
                    few_shot += (f"\n#Example Code {i}:"
                                 f"\n{sim_code}"
                                 f"\n# The comment is: {sim_comment}")
                else:
                    few_shot += (f"\n# Example Code {i}:"
                                 f"\n{sim_code}"
                                 f"\n{prompt_cot_info_augment}"
                                 f"{generate_semantic_tokens4sentence(ids2code_stat_attention_weight, select_ids=[sim_ids])[0]}"
                                 f"\n{prompt_cot_comment}"
                                 f"\n{sim_comment}")
            if example_num <= 0:
                final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], intent_prompt_lists[testcase_intent], testcase_code, intent_prompt_lists[testcase_intent])
            else:
                if llm == "baseline":
                    final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], few_shot, testcase_code)
                else:
                    final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], intent_prompt_lists[testcase_intent], few_shot, testcase_code, intent_prompt_lists[testcase_intent])

            out_path = f"./{dataset}_fix_examples/retrieve_3/{testcase_intent}/{ids}"

            os.makedirs(out_path, exist_ok=True)
            with open(f"{out_path}/{ids}_{llm}_{example_num}shot_{str(is_code_search_attention)}.txt", "w+") as output:
                output.write(final_prompt)
        except Exception as e:
            print(ids)
            print(f"{e}")


def construct_prompts_fusion_quality_sim(selected_testcases_ids=[], dataset="tlcodesum", sim_rule="sim_semantic", intent="all", example_num=3,
                                         is_quality=True, llm='cot', is_code_search_attention="comment2code", mode="train", k=0.8):
    # is_code_search_attention=["comment2code","code2comment",None]
    def sort_normalize(attention_dict, max_line_num=10, min_line_num=3):
        '''
        :param attention_dict:根据正则化后的attention值对其排序
        '''

        def normal(arr):
            # 由于attention数值太少，使用sklearn的tranfer不大行
            res = []
            for x in arr:
                tmp = float(x - np.min(arr)) / (np.max(arr) - np.min(arr))
                res.append(tmp)
            return res

        normal_attention = {j: v for j, v in enumerate(normal(list(attention_dict.values())))}
        sorted_attention = sorted(normal_attention.items(), key=lambda x: x[1], reverse=True)
        line_id = [i[0] for i in sorted_attention]
        percentage = 0.4
        select_token_num = max(min(max_line_num, math.ceil(len(line_id) * percentage)), min_line_num)

        return sorted(line_id[:select_token_num])

    def generate_semantic_tokens4sentence(ids2code_stat_attention_weight, select_ids=[]):
        # semantic: [19495, 71743, 5245]
        # ast: [72116, 37755, 44859]
        def formate_semantic_tokens4sentance(sentance_list):
            res = []
            forbidden_word = ["try {", "else {", "catch ", "finally {", "printStackTrace", "break;", "continue;"]
            for line_number, statement in sentance_list:
                flag = True
                for i in forbidden_word:
                    flag = flag and (i not in statement)
                flag = flag and (statement != '}')

                if flag:
                    if statement.strip() == '':
                        continue
                    elif ";" in statement[-1]:
                        res.append((line_number, statement.split(";")[0]))
                    elif "{" in statement[-1]:
                        res.append((line_number, statement[0:-1]))
                    else:
                        res.append((line_number, statement))
            return res

        # ids2code_stat_attention_weight = {ids: attention_weight for ids, attention_weight in
        #                                   zip(ids2code_stat_format_lines.keys(), code_stat_attention_weight)}
        res = []
        for ids in select_ids:
            tmp = ""

            format_sentence = formate_semantic_tokens4sentance(
                [(int(line_number), ids2code_stat_format_lines[ids][int(line_number)])
                 for line_number in ids2code_stat_attention_weight[ids]])

            for line_id, statement in format_sentence:
                tmp += f"\n{line_id} {statement}"

            res.append(tmp)

        return res

    def priority(sim_traincase_code_ids_score, example_num):
        sim_scores = [sim_score for (sim_ids, sim_score) in sim_traincase_code_ids_score]
        quality_scores = [traincases_quality[sim_ids] for (sim_ids, sim_score) in sim_traincase_code_ids_score]
        sim_rank = np.array(sim_scores).argsort().argsort()
        quality_rank = np.array(quality_scores).argsort().argsort()
        extra = np.array([0 if sim_score < 0.85 else 1000 for (sim_ids, sim_score) in sim_traincase_code_ids_score])
        final_score = k * sim_rank + (1 - k) * quality_rank + extra

        return list(sorted(np.argsort(-np.array(final_score))[:example_num]))

    if example_num <= 0:
        with open(f"prompt_templates/prompt_{llm}_0shot", 'r', encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        with open(f"prompt_templates/prompt_{llm}", 'r', encoding="utf-8") as f:
            prompt_template = f.read()
    with open(f"dataset/{dataset}/train/intents/quality.train", 'r', encoding="utf-8") as f_quality:
        traincases_quality = [item for item in jsonlines.Reader(f_quality)][0]
    with open(f"dataset/{dataset}/test/intents/{intent}.test", "r+", encoding="utf-8") as f_test:
        testcases = {item["ids"]: item for item in jsonlines.Reader(f_test)}
    with open(f"dataset/{dataset}/train/intents/{intent}.train", "r+", encoding="utf-8") as f_train:
        traincases = {item["ids"]: item for item in jsonlines.Reader(f_train)}
    with open(f"dataset/{dataset}/test/intents/{sim_rule}/{sim_rule}_score.{intent}", "r+", encoding="utf8") as f_sim:
        sim_traincases_ids = {item["ids"]: item["sim_code"] for item in jsonlines.Reader(f_sim)}
    with open(f'./dataset/{dataset}/{mode}/code_split_format.{mode}', 'r') as f:  # 对code分行进行格式处理
        ids2code_stat_format_lines = {item["ids"]: item["code"] for item in jsonlines.Reader(f)}

    code_stat_attention_weight = None
    if is_code_search_attention == "comment2code":  # 读取模型attention信息
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight-cs.{mode}', 'r') as f:
            code_stat_attention_weight = [sort_normalize(item) for item in jsonlines.Reader(f)]
    elif is_code_search_attention == "code2comment":
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight-cs-20_code2comment.{mode}', 'r') as f:
            code_stat_attention_weight = [sort_normalize(item) for item in jsonlines.Reader(f)]
    else:
        with open(f'dataset/{dataset}/{mode}/attentions/code_stat_attention_weight_-1.{mode}', 'r') as f:
            code_stat_attention_weight = [list(item.keys()) for item in jsonlines.Reader(f)]

    ids2code_stat_attention_weight = {ids: attention_weight for ids, attention_weight in
                                      zip(ids2code_stat_format_lines.keys(), code_stat_attention_weight)}
    # 选择提取目标
    selected_testcases_ids = list(sim_traincases_ids.keys()) if len(selected_testcases_ids) == 0 else selected_testcases_ids
    print(len(selected_testcases_ids))

    for ids in tqdm(selected_testcases_ids):

        try:
            testcase = testcases[ids]
            testcase_code = testcase["raw_code"]
            testcase_comment = testcase["comment"]
            testcase_intent = testcase["label"]

            few_shot = ""

            sim_traincase_code_ids_score = sim_traincases_ids[ids]  # 选择相似example
            if is_quality:
                # fusion_example_score = [k * sim_score + (1 - k) * traincases_quality[sim_ids] if sim_score < 0.85 else 1 for (sim_ids, sim_score) in sim_traincase_code_ids_score]
                # select = list(sorted(np.argsort(-np.array(fusion_example_score))[:example_num]))
                select = priority(sim_traincase_code_ids_score, example_num)

            else:
                select = [i for i in range(example_num)]

            for i, idx in enumerate(select[:example_num]):
                (sim_ids, sim_score) = sim_traincase_code_ids_score[idx]
                # 构造prompt
                sim_code = traincases[sim_ids]["raw_code"]
                sim_comment = traincases[sim_ids]["comment"]

                if llm == "baseline":
                    few_shot += (f"\n#Example Code {i}:"
                                 f"\n{sim_code}"
                                 f"\n# The comment is: {sim_comment}")
                else:
                    few_shot += (f"\n# Example Code {i}:"
                                 f"\n{sim_code}"
                                 f"\n{prompt_cot_info_augment}"
                                 f"{generate_semantic_tokens4sentence(ids2code_stat_attention_weight, select_ids=[sim_ids])[0]}"
                                 f"\n{prompt_cot_comment}"
                                 f"\n{sim_comment}")
            if example_num <= 0:
                final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], intent_prompt_lists[testcase_intent], testcase_code, intent_prompt_lists[testcase_intent])
            else:
                if llm == "baseline":
                    final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], few_shot, testcase_code)
                else:
                    final_prompt = prompt_template.format(intent_prompt_lists[testcase_intent], intent_prompt_lists[testcase_intent], few_shot, testcase_code, intent_prompt_lists[testcase_intent])

            if is_code_search_attention != "comment2code":
                out_path = f"./{dataset}_{is_code_search_attention}/retrieve_{sim_rule}/{testcase_intent}/{ids}"
            elif is_quality and llm == 'baseline':
                out_path = f"./{dataset}_no_cot/retrieve_{sim_rule}/{testcase_intent}/{ids}"
            elif is_quality:
                out_path = f"./{dataset}/retrieve_{sim_rule}/{testcase_intent}/{ids}"
            else:
                out_path = f"./{dataset}_no_quality/retrieve_{sim_rule}/{testcase_intent}/{ids}"

            os.makedirs(out_path, exist_ok=True)
            with open(f"{out_path}/{ids}_{llm}_{example_num}shot_{str(is_code_search_attention)}.txt", "w+") as output:
                output.write(final_prompt)
        except Exception as e:
            print(ids)
            print(f"{e}")

def generate_samples():
    dataset = 'tlcodesum'
    sim_rule = 'sim_semantic'
    is_quality = True

    construct_prompts(dataset=dataset, selected_testcases_ids=[], example_num=0, sim_rule=sim_rule, is_quality=True, llm='cot',
                      is_code_search_attention='comment2code')
    construct_prompts(dataset=dataset, selected_testcases_ids=[], example_num=3, sim_rule=sim_rule, is_quality=True, llm='cot',
                      is_code_search_attention='comment2code')
    construct_prompts(dataset=dataset, selected_testcases_ids=[], example_num=5, sim_rule=sim_rule, is_quality=True, llm='cot',
                      is_code_search_attention='comment2code')
    construct_prompts(dataset=dataset, selected_testcases_ids=[], example_num=0, sim_rule=sim_rule, is_quality=False, llm='baseline',
                      is_code_search_attention='comment2code')
    construct_prompts(dataset=dataset, selected_testcases_ids=[], example_num=3, sim_rule=sim_rule, is_quality=False, llm='baseline',
                      is_code_search_attention='comment2code')
    construct_prompts(dataset=dataset, selected_testcases_ids=[], example_num=5, sim_rule=sim_rule, is_quality=False, llm='baseline',
                      is_code_search_attention='comment2code')
    construct_prompts(dataset=dataset, selected_testcases_ids=[], example_num=10, sim_rule=sim_rule, is_quality=False, llm='baseline',
                      is_code_search_attention='comment2code')

    # print(ids)


if __name__ == '__main__':
    generate_samples()
