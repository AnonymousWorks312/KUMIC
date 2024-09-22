import json
import os

import jsonlines
import math
import nltk
import numpy as np
from nltk import WordNetLemmatizer
from scipy import spatial
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

import evaluation_utils
from pandas import DataFrame
import pandas as pd
import shutil

prompt_types = {'all': ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
                        'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
                        'cot_0shot_comment2code', 'cot_3shot_comment2code', 'cot_5shot_comment2code'],
                'all_no_zero': ['baseline_3shot_comment2code', 'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
                                'cot_3shot_comment2code', 'cot_5shot_comment2code'],
                'cot': ['cot_0shot_comment2code', 'cot_3shot_comment2code', 'cot_5shot_comment2code'],
                'cot_no_zero': ['cot_3shot_comment2code', 'cot_5shot_comment2code'],
                'no_cot': ['cot_3shot_comment2code'],
                'cg': ['cot_3shot_cg'],
                'baseline': ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
                             'baseline_5shot_comment2code', 'baseline_10shot_comment2code'],
                'fix': ['baseline_3shot_comment2code', 'cot_3shot_comment2code'],
                'one': ['cot_3shot_comment2code']
                }



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


def print_evaluation(name, target, inference):
    print(f"=============={name}===============")
    evaluation_utils.calculate_all(target.lower(), inference.lower())


def print_evaluation4batch(name, target, inferences):
    for idx, inference in enumerate(inferences):
        print_evaluation(f"{name}-{idx}", target, inference)


# 数据是分开存储的
def save_result2execl(prompt_types, path="retrieve_sim_token", dataset='tlcodesum', date='0619', llm='llama3',
                      remove_duplicate=True):
    # prompt_types = ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
    #                 'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
    #                 'cot_0shot_comment2code',
    #                 'cot_3shot_comment2code', 'cot_5shot_comment2code']
    intents = ['what', 'why', 'done', 'property', 'usage']

    baseline_path = None
    if dataset == 'tlcodesum' and 'sim_semantic' in path:
        if llm == 'codellama':
            baseline_path = f"../tlcodesum_baseline/retrieve_sim_semantic/reformat_comment-0531.pred"
        elif llm == 'llama3':
            baseline_path = f"../tlcodesum_baseline/retrieve_sim_semantic/reformat_comment-0623.pred"
    elif dataset == 'tlcodesum' and 'sim_token' in path:
        if llm == 'codellama':
            baseline_path = f"../tlcodesum_baseline/retrieve_sim_token/reformat_comment-0619.pred"
        elif llm == 'llama3':
            baseline_path = f"../tlcodesum/retrieve_sim_token/reformat_comment-0629.pred"
    if dataset == 'funcom' and 'sim_semantic' in path:
        if llm == 'codellama':
            baseline_path = f"../funcom/retrieve_sim_semantic/reformat_comment-0627.pred"
        elif llm == 'llama3':
            baseline_path = f"../funcom/retrieve_sim_semantic/reformat_comment-0629.pred"
    elif dataset == 'funcom' and 'sim_token' in path:
        if llm == 'codellama':
            baseline_path = f"../funcom/retrieve_sim_token/reformat_comment-0630.pred"
        elif llm == 'llama3':
            baseline_path = f"../funcom/retrieve_sim_token/reformat_comment-0629.pred"
    if baseline_path is not None:
        with open(baseline_path, 'r') as f:
            intents_comment = json.loads(f.read())
    else:
        with open('../tlcodesum_baseline/retrieve_sim_semantic/reformat_comment-0531.pred', 'r') as f:
            intents_comment = json.loads(f.read())

    with open(f"{path}/reformat_comment-{date}.pred", 'r') as f:
        print(f"{path}/reformat_comment-{date}.pred")
        intents_comment_cot = json.loads(f.read())
    with open(f"../dataset/{dataset}/test/intents/all.test", 'r') as f:
        testcases = {}
        ids_order = []
        for item in jsonlines.Reader(f):
            testcases[item['ids']] = item
            ids_order.append(item['ids'])
    with open(f"../dataset/{dataset}/test/same.test", "r") as f_same:
        same = [i.strip() for i in f_same.readlines()]
    with open(f"../results/{dataset}/DOME", 'r') as f:
        dome_results = {ids_order[i]: comment for i, comment in enumerate(f.readlines())}
    print("Data loading finished")

    titles = ["ids", "intent", "ground_truth", "dome"] + prompt_types
    methods = ['dome'] + prompt_types
    sheet_names = ['bleu',
                   'meteor',
                   'rough-l', ]
    sheets_data = {sheet_name: {title: [] for title in titles} for sheet_name in sheet_names}
    comment_data = {title: [] for title in titles}
    saved_data = {title: [] for title in titles}

    def load_comment(intents_comment):
        # 将comment数据加载到execl
        for intent in intents:
            print(f"in {intent}")
            ids_llmcomment = intents_comment[intent]
            ids_llmcomment_cot = intents_comment_cot[intent]

            for ids in tqdm(ids_llmcomment_cot.keys()):
                comment_data["ids"].append(ids)
                comment_data["intent"].append(intent)
                comment_data["ground_truth"].append(reformat_comment(testcases[ids]["comment"]))
                comment_data["dome"].append(dome_results[ids])

                cot_keys = ids_llmcomment_cot[ids].keys()

                for i in prompt_types:
                    if i in cot_keys:
                        comment_data[i].append([comment[:-1] if dataset == 'funcom' and comment[-1] == '.' and len(comment[:-1]) > 0 else comment for comment in ids_llmcomment_cot[ids][i]])  # 这个comment序列
                    else:
                        comment_data[i].append([comment[:-1] if dataset == 'funcom' and comment[-1] == '.' and len(comment[:-1]) > 0 else comment for comment in ids_llmcomment[ids][i]])  # 这个comment序列

    def get_element(intent, method, bleu_scores, meteor_scores, rough_scores):
        if len(bleu_scores) == 0:
            print('find zero')
            return 0, 0, 0, 0
        else:
            idx = np.argsort(bleu_scores)[-1]

        return idx, np.array(bleu_scores)[idx], np.array(meteor_scores)[idx], np.array(rough_scores)[idx]
        # return list(np.array(bleu_scores)[idx]), list(np.array(meteor_scores)[idx]), list(np.array(rough_scores)[idx])

    def cal_meteor4batch():
        if os.path.exists(f"{path}/cached_meteor_score-{date}.pred"):
            with open(f"{path}/cached_meteor_score-{date}.pred", 'r') as f:
                res = json.loads(f.read())
            return res

        res = {}
        for method in tqdm(methods):
            comment_ref = []
            comment_pred = []
            idx_list = []  # 记录每个ids对应comment数量的数组
            if method == 'dome':
                for i in range(len(comment_data['ids'])):
                    comment_num = 1
                    comment_ref.append(comment_data['ground_truth'][i])
                    comment_pred.append(comment_data[method][i])
                    idx_list.append(comment_num)
            else:
                for i in range(len(comment_data['ids'])):
                    comment_num = len(comment_data[method][i])
                    for j in range(comment_num):
                        comment_ref.append(comment_data['ground_truth'][i])
                        comment_pred.append(comment_data[method][i][j])
                    idx_list.append(comment_num)
            print(len(comment_ref))
            meteor_scores = evaluation_utils.cal_meteor_new(comment_pred, comment_ref)
            res[method] = {}
            sum_idx = 0

            for i, idx in enumerate(idx_list):
                res[method][str(i)] = meteor_scores[sum_idx:sum_idx + idx]
                sum_idx += idx
        with open(f"{path}/cached_meteor_score-{date}.pred", 'w+') as f:
            json.dump(res, f)

        return res

    load_comment(intents_comment)
    meteor_score_dict = cal_meteor4batch()

    # 对于每个case计算各种方法的bleu、meteor、rough-L_f1得分
    for i in tqdm(range(len(comment_data['ids']))):
        ground_truth = comment_data["ground_truth"][i]
        intent = comment_data["intent"][i]
        ids = comment_data["ids"][i]
        if remove_duplicate and ids in same:
            continue

        saved_data["ids"].append(ids)
        saved_data["intent"].append(intent)
        saved_data["ground_truth"].append(ground_truth)

        for method in methods:
            if method == 'dome':
                method_comment = [comment_data[method][i]]
            else:
                method_comment = comment_data[method][i]
            # print(f'{ground_truth=}')
            # print(f'{method_comment=}')
            bleu_scores, rough_scores = evaluation_utils.calculate_all4list_new(ground_truth, method_comment)

            meteor_scores = meteor_score_dict[method][str(i)]
            idx, bleu_scores, meteor_scores, rough_scores = get_element(intent, method, bleu_scores, meteor_scores, rough_scores)

            sheets_data['bleu'][method].append(bleu_scores)
            sheets_data['meteor'][method].append(meteor_scores)
            sheets_data['rough-l'][method].append(rough_scores)
            if idx < len(method_comment):
                saved_data[method].append(method_comment[idx])
            else:
                saved_data[method].append('None')

        for sheet_name in sheets_data.keys():
            sheets_data[sheet_name]["ids"].append(ids)
            sheets_data[sheet_name]["intent"].append(intent)
            sheets_data[sheet_name]["ground_truth"].append(1)
    # 计算平均值
    for sheet_name in sheets_data.keys():
        sheets_data[sheet_name]["ids"].append('Average')
        sheets_data[sheet_name]["intent"].append('None')
        sheets_data[sheet_name]["ground_truth"].append(1)

        for method in methods:
            tmp = sheets_data[sheet_name][method]
            sheets_data[sheet_name][method].append(np.mean(tmp))

    # with open(f'data-{date}-meteorsample-better-withintent-40.test', 'w') as f:
    #     json.dump(saved_data, f)

    # with pd.ExcelWriter(f'evaluation-{path.split("/")[6]}-{path.split("/")[7]}-{date}-{llm}_trick-7.xlsx') as writer:
    if remove_duplicate:
        res_file_name = f'evaluation-{path.split("/")[6]}-{path.split("/")[7]}-{date}-{llm}-remove_duplicate.xlsx'
    else:
        res_file_name = f'evaluation-{path.split("/")[6]}-{path.split("/")[7]}-{date}-{llm}.xlsx'
    with pd.ExcelWriter(f'{res_file_name}') as writer:
        DataFrame(saved_data).to_excel(writer, sheet_name='comment', index=False)
        for sheet_name in sheets_data.keys():
            DataFrame(sheets_data[sheet_name]).to_excel(writer, sheet_name, index=False)
    # print(f'{delete_num=}')



def read_excel_data4intents(execl_path, prompt_types, dataset):
    # prompt_types = ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
    #                 'baseline_5shot_comment2code', 'baseline_10shot_comment2code',
    #                 'cot_0shot_comment2code', 'cot_3shot_code2comment',
    #                 'cot_3shot_comment2code', 'cot_5shot_comment2code']
    # prompt_types = ['baseline_0shot_comment2code', 'baseline_3shot_comment2code',
    #                 'baseline_5shot_comment2code',
    #                 'cot_0shot_comment2code',
    #                 'cot_3shot_comment2code', 'cot_5shot_comment2code']
    methods = ['dome'] + prompt_types
    # execl_path = 'evaluation-0619-bleusample-better-final.xlsx'
    df = pd.read_excel(execl_path, sheet_name=None)
    sheet_names = list(df.keys())
    titles = ["ids", "ground_truth"] + methods
    ids_list = list(df['bleu']['ids'])

    with open(f"../{dataset}/retrieve_sim_semantic/reformat_comment-0629.pred", 'r') as f:
        intents_ids = {key: list(value.keys()) for key, value in json.loads(f.read()).items()}

    for intent in intents_ids.keys():
        sheets_data = {sheet_name: {title: [] for title in titles if title != 'intent'} for sheet_name in sheet_names}
        print(intent)
        for ids in tqdm(intents_ids[intent]):
            try:
                idx = ids_list.index(ids)
                for sheet_name in sheet_names:
                    for title in titles:
                        if title == 'intent':
                            continue
                        sheets_data[sheet_name][title].append(df[sheet_name][title][idx])
            except ValueError:
                print('not id')

        for sheet_name in sheets_data.keys():
            if sheet_name == 'comment':
                continue
            sheets_data[sheet_name]["ids"].append('Average')
            sheets_data[sheet_name]["ground_truth"].append(1)

            for method in methods:
                tmp = sheets_data[sheet_name][method]
                sheets_data[sheet_name][method].append(np.mean(tmp))

        path = f'{execl_path.split(".")[0]}'
        os.makedirs(path, exist_ok=True)
        with pd.ExcelWriter(f'{path}/{intent}.xlsx') as writer:
            # DataFrame(comment_data).to_excel(writer, sheet_name='comment', index=False)
            for sheet_name in sheet_names:
                DataFrame(sheets_data[sheet_name]).to_excel(writer, sheet_name, index=False)


def sample_manual(execl_path):
    prompt_types = ['baseline_5shot_comment2code',
                    'cot_3shot_comment2code']

    df = pd.read_excel(execl_path, sheet_name=None)
    datasets = list(df.keys())
    titles = ["ids", "intent", "ground_truth", "dome"] + prompt_types
    methods = ['dome'] + prompt_types
    res = {title: [] for title in titles}
    code_file_name = f'../samples/code'

    def preprocess():
        for dataset in datasets:
            if dataset == 'funcom':
                df_all = pd.read_excel('evaluation-funcom-retrieve_sim_semantic-0627-codellama-remove_duplicate.xlsx', sheet_name='comment')
            else:
                df_all = pd.read_excel('evaluation-tlcodesum-retrieve_sim_semantic-0626-codellama.xlsx', sheet_name='comment')
            # with open(f"../dataset/{dataset}/test/intents/all.test", 'r') as f:
            #     testcases = {}
            #     ids_order = []
            #     for item in jsonlines.Reader(f):
            #         testcases[item['ids']] = item
            #         ids_order.append(item['ids'])
            # with open(f"../results/{dataset}/DOME", 'r') as f:
            #     dome_results = {ids_order[i]: comment for i, comment in enumerate(f.readlines())}

            # sheets_data = {sheet_name: {title: [] for title in titles} for sheet_name in sheet_names}

            comment_data = {title: [] for title in titles}
            with open(f"../dataset/{dataset}/test/intents/all.test", 'r') as f:
                testcases = {}
                for item in jsonlines.Reader(f):
                    testcases[item['ids']] = item
            for ids in tqdm(list(df[dataset]['ids'])):
                index_in_all = list(df_all['ids']).index(ids)
                for title in titles:
                    comment_data[title].append(df_all[title][index_in_all])

            for i in tqdm(range(len(comment_data['ids']))):
                ground_truth = comment_data["ground_truth"][i]
                intent = comment_data["intent"][i]
                ids = comment_data["ids"][i]
                tmp = {}
                for method in methods:
                    method_comment = [comment_data[method][i]]
                    bleu_scores, rough_scores = evaluation_utils.calculate_all4list_new(ground_truth, method_comment)

                    tmp[method] = bleu_scores[0]
                if tmp['dome'] < tmp['cot_3shot_comment2code'] and tmp['baseline_5shot_comment2code'] < tmp['cot_3shot_comment2code']:
                    for title in titles:
                        res[title].append(comment_data[title][i])


                    os.makedirs(code_file_name, exist_ok=True)
                    with open(f"{code_file_name}/{res['ids'][-1]}.txt", 'w+') as f:
                        f.write(testcases[str(res['ids'][-1])]['raw_code'])

    def random_selection(num=4):
        total = 100
        comment_titles = ["ids", "intent", "dome"] + prompt_types
        for i in range(num):
            import random
            random_ids = random.sample(list(res['ids']), int(total/num))
            random_file_name = f'../samples/vol_{i}'
            os.makedirs(random_file_name, exist_ok=True)
            tmp = {title: [] for title in comment_titles}
            for ids in sorted(random_ids):
                shutil.copy(f'{code_file_name}/{ids}.txt', f'{random_file_name}')
                index = res['ids'].index(ids)
                for title in comment_titles:
                    tmp[title].append(res[title][index])

            with pd.ExcelWriter(f'{random_file_name}/comments.xlsx') as writer:
                DataFrame(tmp).to_excel(writer, sheet_name='comment', index=False)



    preprocess()
    random_selection()

    res_file_name = f'../samples/manual.xlsx'

    with pd.ExcelWriter(f'{res_file_name}') as writer:
        DataFrame(res).to_excel(writer, sheet_name='comment', index=False)


if __name__ == '__main__':

    path = ['./tlcodesum_fix_examples/retrieve_1',
            './tlcodesum_fix_examples/retrieve_2',
            './tlcodesum_fix_examples/retrieve_3',
            './tlcodesum_0.5/retrieve_sim_semantic',
            './tlcodesum_0.75/retrieve_sim_semantic',
            './tlcodesum_cg_0.5/retrieve_sim_semantic',
            './tlcodesum_cg_0.75/retrieve_sim_semantic'
            ]

    prompt_type4path = ['fix',
                        'fix',
                        'fix',
                        'one',
                        'one',
                        'one',
                        'one']

    datasets = ['tlcodesum', 'tlcodesum', 'tlcodesum', 'tlcodesum',
                'tlcodesum', 'tlcodesum','tlcodesum', 'tlcodesum']

    idxs = [5]
    for i in idxs:
        save_result2execl(prompt_types=prompt_types[prompt_type4path[i]],
                      path=path[i],
                      dataset=datasets[i],
                      date='0910',
                      llm='codellama',
                      remove_duplicate=True)
        save_result2execl(prompt_types=prompt_types[prompt_type4path[i]],
                          path=path[i],
                          dataset=datasets[i],
                          date='0909',
                          llm='llama3',
                          remove_duplicate=True)

    # find_low(path='./funcom/retrieve_sim_semantic',
    #                       dataset='funcom',
    #                       date='0629',
    #                       llm='llama3',
    #                       remove_duplicate=True)

    # merge_zero_shot()
    # exit(0)
    # save_result2execl4old(path='./tlcodesum_no_quality/retrieve_sim_semantic',
    #                       dataset='tlcodesum',
    #                       date='0629',
    #                       llm='llama3')
    # save_result2execl4old(path='./tlcodesum_no_quality/retrieve_sim_token',
    #                       dataset='tlcodesum',
    #                       date='0629',
    #                       llm='llama3')
    # save_result2execl4one(path='./tlcodesum_fusion_quality/retrieve_sim_semantic',
    #                       date='0625')
    # read_excel_data()
    # evaluate_quality()
    # save_result2execl4intent()
