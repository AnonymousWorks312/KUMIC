import json
import os
import re
import time
import random
from threading import Thread
from typing import NoReturn, Callable

import numpy as np
from jsonlines import jsonlines
from matplotlib import pyplot as plt
from scipy import spatial
# from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# from java_ast_parser import fee


def store_data_from_command(out_path, file_name, store_command: Callable):
    '''
    创建文件夹，并存储文件
    :param out_path:文件存储路径
    :param file_name:文件名
    :param store_command:用于存储文件的命令
    '''
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("Folder created")

    with open(f"{out_path}/{file_name}", "w+") as f:
        store_command(f)


def code_sentance_split(raw_code_path, code_split_path, code_split_format_path, code_split_format_error_id_path):
    with open(raw_code_path, 'r') as f, \
            open(code_split_path, 'r') as f_split, \
            open(code_split_format_path, 'w+') as format_f_split:

        test_set = [item for item in jsonlines.Reader(f)]
        test_set_split = [item for item in jsonlines.Reader(f_split)]
        error_ids_list = []
        for index in range(len(test_set)):
            result = []
            raw_code_lines = [line.strip() for line in test_set[index]["raw_code"].splitlines() if line]

            i = j = 0

            while i < len(raw_code_lines) and j < len(test_set_split[index]["code"]):
                first_token = test_set_split[index]["code"][j].split()[0]
                if first_token in raw_code_lines[i] or first_token[0:-4] in raw_code_lines[i]:
                    result.append(raw_code_lines[i])
                    i += 1
                    j += 1
                else:
                    i += 1
            if len(test_set_split[index]["code"]) == len(result):
                test_set_split[index]["code"] = result
            else:
                error_ids_list.append(test_set[index]["ids"])

            format_f_split.write(json.dumps(test_set_split[index]) + "\n")

        if len(error_ids_list) != 0:
            with open(code_split_format_error_id_path, 'w+') as format_f_split_error:
                for ids in error_ids_list:
                    format_f_split_error.write(str(ids) + "\n")


def category_code4intents(dataset="tlcodesum", mode="train"):
    print(f'[intents] dataset={dataset} mode={mode}')
    with open(f"dataset/{dataset}/{mode}/raw_code.{mode}", "r") as f_raw, \
            open(f"dataset/{dataset}/{mode}/code_split.{mode}", "r") as f_spl, \
            open(f"dataset/{dataset}/{mode}/code.{mode}", "r") as f_code, \
            open(f"dataset/{dataset}/{mode}/comment.{mode}", "r") as f_comment, \
            open(f"dataset/{dataset}/{mode}/label.{mode}", "r") as f_label:
        raw_code = [item for item in jsonlines.Reader(f_raw)]
        spl_code = [item for item in jsonlines.Reader(f_spl)]
        code = [item for item in f_code.read().splitlines()]  # code分词后存储为当行数据
        comment = [item for item in f_comment.read().splitlines()]
        label = [item for item in f_label.read().splitlines()]
        print(len(raw_code))
        print(len(code))
        print(len(spl_code))
        print(len(comment))
        print(len(label))

        intents = ["what", "why", "done", "usage", "property"]
        intents_res = {i: "" for i in intents}

        for i in tqdm(range(len(raw_code))):
            data = {}
            data["ids"] = raw_code[i]["ids"]
            data["raw_code"] = raw_code[i]["raw_code"]
            data["spl_code"] = spl_code[i]["code"]
            # data["ast"] = fee(data["raw_code"])  # ast[i]
            data["code"] = code[i]
            data["comment"] = comment[i]
            data["label"] = label[i]
            intents_res[data["label"]] += (json.dumps(data) + "\n")

        for intent in intents:
            store_data_from_command(out_path=f"dataset/{dataset}/{mode}/intents",
                                    file_name=f"{intent}.{mode}",
                                    store_command=lambda file: file.write(intents_res[intent]))


def category_code4all(dataset="tlcodesum", mode="train"):
    print(f'[all] dataset={dataset} mode={mode}')
    with open(f"dataset/{dataset}/{mode}/raw_code.{mode}", "r") as f_raw, \
            open(f"dataset/{dataset}/{mode}/code_split.{mode}", "r") as f_spl, \
            open(f"dataset/{dataset}/{mode}/code.{mode}", "r") as f_code, \
            open(f"dataset/{dataset}/{mode}/comment.{mode}", "r") as f_comment, \
            open(f"dataset/{dataset}/{mode}/label.{mode}", "r") as f_label:
        raw_code = [item for item in jsonlines.Reader(f_raw)]
        spl_code = [item for item in jsonlines.Reader(f_spl)]
        code = [item for item in f_code.read().splitlines()]  # code分词后存储为当行数据
        comment = [item for item in f_comment.read().splitlines()]
        label = [item for item in f_label.read().splitlines()]
        print(len(raw_code))
        print(len(code))
        print(len(spl_code))
        print(len(comment))
        print(len(label))
        res = ""

        for i in tqdm(range(len(raw_code))):
            data = {}
            data["ids"] = raw_code[i]["ids"]
            data["raw_code"] = raw_code[i]["raw_code"]
            data["spl_code"] = spl_code[i]["code"]
            # data["ast"] = fee(data["raw_code"])  # ast[i]
            data["code"] = code[i]
            data["comment"] = comment[i]
            data["label"] = label[i]
            res += (json.dumps(data) + "\n")

        store_data_from_command(out_path=f"dataset/{dataset}/{mode}/intents",
                                file_name=f"all.{mode}",
                                store_command=lambda file: file.write(res))


def category_code4intent_classifier(dataset="tlcodesum", mode="train"):
    with open(f"dataset/{dataset}/{mode}/raw_code.{mode}", "r") as f_raw, \
            open(f"dataset/{dataset}/{mode}/code_split.{mode}", "r") as f_spl, \
            open(f"dataset/{dataset}/{mode}/code.{mode}", "r") as f_code, \
            open(f"dataset/{dataset}/{mode}/comment.{mode}", "r") as f_comment, \
            open(f"dataset/{dataset}/{mode}/label.{mode}", "r") as f_label:
        raw_code = [item for item in jsonlines.Reader(f_raw)]
        spl_code = [item for item in jsonlines.Reader(f_spl)]
        code = [item for item in f_code.read().splitlines()]  # code分词后存储为当行数据
        comment = [item for item in f_comment.read().splitlines()]
        label = [item for item in f_label.read().splitlines()]
        print(len(raw_code))
        print(len(code))
        print(len(spl_code))
        print(len(comment))
        print(len(label))

        output_dict = {'id': [], 'code': [], 'comment': [], 'label': []}

        for i in tqdm(range(len(raw_code))):
            output_dict['id'].append(int(raw_code[i]["ids"]))
            output_dict['code'].append(code[i])
            output_dict['comment'].append(comment[i])
            output_dict['label'].append(label[i])

        store_data_from_command(out_path=f"dataset/{dataset}/{mode}",
                                file_name=f"{dataset}_comment_data_final.{mode}",
                                store_command=lambda file: json.dump(output_dict, file))
        # with open( / {dataset}_comment_data_final.{mode}", "w+") as f_all:
        # json.dump(output_dict, f_all)


def category_code4code_search_comment2code(dataset="tlcodesum", mode="train", neg_samples_num=20, trainingset_percentage:float=1):
    with open(f"dataset/{dataset}/{mode}/intents/all.{mode}") as f_all:
        traincases = [item for item in jsonlines.Reader(f_all)]

    res = ""
    traincases = random.sample(traincases, int(len(traincases) * trainingset_percentage))
    for i in tqdm(range(len(traincases))):
        comment_ids = traincases[i]["ids"]
        comment = traincases[i]["comment"]
        comment_intent = traincases[i]["label"]

        # 随机选择负样本
        negative_samples_ids = random.sample([x for x in range(len(traincases)) if x != i], neg_samples_num)
        # 构造负样本+正样本
        for j in [i] + negative_samples_ids:
            '''
            code search 数据格式：
            1
            <CODESPLIT>{"comment_ids":"12345", "code_ids": "12345"}
            <CODESPLIT>{"comment_intent":"what", "code_intent": "done"}
            <CODESPLIT>comment
            <CODESPLIT>{"code":code, "spl_code":[spl_code]}
            '''
            code_ids = traincases[j]["ids"]
            code_intent = traincases[j]["label"]
            comment_code_ids_pair = {"comment_ids": comment_ids, "code_ids": code_ids}
            comment_code_intent_pair = {"comment_intent": comment_intent, "code_intent": code_intent}
            code_info = {"code": traincases[j]["code"], "spl_code": traincases[j]["spl_code"]}
            data = (f"{int(comment_ids == code_ids)}"
                    f"<CODESPLIT>{json.dumps(comment_code_ids_pair)}"
                    f"<CODESPLIT>{json.dumps(comment_code_intent_pair)}"
                    f"<CODESPLIT>{comment}"
                    f"<CODESPLIT>{json.dumps(code_info)}")
            res += (data + "\n")
        # print(res)

    store_data_from_command(out_path=f"dataset/code_search/{dataset}",
                            file_name=f"{dataset}-{neg_samples_num}-spl-code-comment2code-{trainingset_percentage}.{mode}",
                            store_command=lambda file: file.write(res))

def category_code4code_search_comment2code2(dataset="tlcodesum", mode="train"):
    with open(f"dataset/{dataset}/{mode}/intents/all.{mode}") as f_all:
        traincases = [item for item in jsonlines.Reader(f_all)]

    res = ""

    for i in tqdm(range(len(traincases))):
        comment_ids = traincases[i]["ids"]
        comment = traincases[i]["comment"]
        comment_intent = traincases[i]["label"]
        j = i
        # 构造负样本+正样本

        '''
        code search 数据格式：
        1
        <CODESPLIT>{"comment_ids":"12345", "code_ids": "12345"}
        <CODESPLIT>{"comment_intent":"what", "code_intent": "done"}
        <CODESPLIT>comment
        <CODESPLIT>{"code":code, "spl_code":[spl_code]}
        '''
        code_ids = traincases[j]["ids"]
        code_intent = traincases[j]["label"]
        comment_code_ids_pair = {"comment_ids": comment_ids, "code_ids": code_ids}
        comment_code_intent_pair = {"comment_intent": comment_intent, "code_intent": code_intent}
        code_info = {"code": traincases[j]["code"], "spl_code": traincases[j]["spl_code"]}
        data = (f"{int(comment_ids == code_ids)}"
                    f"<CODESPLIT>{json.dumps(comment_code_ids_pair)}"
                    f"<CODESPLIT>{json.dumps(comment_code_intent_pair)}"
                    f"<CODESPLIT>{comment}"
                    f"<CODESPLIT>{json.dumps(code_info)}")
        res += (data + "\n")
        # print(res)

    store_data_from_command(out_path=f"dataset/code_search/{dataset}",
                            file_name=f"{dataset}-0-spl-code-comment2code.{mode}",
                            store_command=lambda file: file.write(res))

def category_code4code_search_code2comment(dataset="tlcodesum", mode="train", neg_samples_num=20):
    with open(f"dataset/{dataset}/{mode}/intents/all.{mode}") as f_all:
        traincases = [item for item in jsonlines.Reader(f_all)]

    res = ""

    for i in tqdm(range(len(traincases))):
        code_ids = traincases[i]["ids"]
        code_info = {"code": traincases[i]["code"], "spl_code": traincases[i]["spl_code"]}
        code_intent = traincases[i]["label"]

        # 随机选择负样本
        negative_samples_ids = random.sample([x for x in range(len(traincases)) if x != i], neg_samples_num)
        # 构造负样本+正样本
        for j in [i] + negative_samples_ids:
            '''
            code search 数据格式：
            1
            <CODESPLIT>{"code_ids":"12345", "comment_ids": "12345"}
            <CODESPLIT>{"code_intent":"what", "comment_intent": "done"}
            <CODESPLIT>comment
            <CODESPLIT>{"code":code, "spl_code":[spl_code]}
            
            '''
            comment_ids = traincases[j]["ids"]
            comment = traincases[j]["comment"]
            comment_intent = traincases[j]["label"]
            comment_code_ids_pair = {"comment_ids": comment_ids, "code_ids": code_ids}
            comment_code_intent_pair = {"comment_intent": comment_intent, "code_intent": code_intent}
            data = (f"{int(comment_ids == code_ids)}"
                    f"<CODESPLIT>{json.dumps(comment_code_ids_pair)}"
                    f"<CODESPLIT>{json.dumps(comment_code_intent_pair)}"
                    f"<CODESPLIT>{comment}"
                    f"<CODESPLIT>{json.dumps(code_info)}")
            res += (data + "\n")
        # print(res)

    store_data_from_command(out_path=f"dataset/code_search/{dataset}",
                            file_name=f"{dataset}-{neg_samples_num}-spl-code--code2comment.{mode}",
                            store_command=lambda file: file.write(res))


def tokenize(code_str):
    code_str = str(code_str)
    code_str = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', code_str)
    code_str = re.sub(r'[\.\,\;\:\(\)\{\}\[\]]', ' ', code_str)
    code_str = re.sub(r'\s+', ' ', code_str)
    tokens = re.findall(r'[a-z]+|[A-Z][a-z]*|[0-9]+|[^\w\s]+', code_str)
    for i in range(len(tokens)):
        if i > 0 and tokens[i - 1].islower() and tokens[i].isupper():
            tokens[i] = tokens[i].lower()
    return tokens


def preprocess_samples_token(intent, dataset="tlcodesum"):
    def count_common_elements(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        common_elements = set1.intersection(set2)
        return len(common_elements)

    testcases = []
    traincases = []
    res = ""
    with open(f"dataset/{dataset}/train/intents/{intent}.train", "r") as f_train, \
            open(f"dataset/{dataset}/test/intents/{intent}.test", "r") as f_test:
        testcases = [item for item in jsonlines.Reader(f_test)]
        traincases = [item for item in jsonlines.Reader(f_train)]
    test_codes_embeddings, training_codes_embeddings = [], []
    st = time.time()
    print('Test code embedding generation start...')
    for i in tqdm(range(len(testcases))):
        test_code = testcases[i]["code"]
        code1_emb = tokenize(test_code)
        test_codes_embeddings.append(code1_emb)
    ed = time.time()
    print('Test code embedding generate finish!')
    print(str(ed - st))
    print('Training code embedding generation start...')
    for i in tqdm(range(len(traincases))):
        train_code = traincases[i]["code"]
        code1_emb = tokenize(train_code)
        training_codes_embeddings.append(code1_emb)
    print('Training code embedding generate finish!')

    for i in tqdm(range(len(testcases))):
        test_code_embedding = test_codes_embeddings[i]
        sim_scores = []
        for j in range(len(traincases)):
            train_code_embedding = training_codes_embeddings[j]
            score = count_common_elements(test_code_embedding, train_code_embedding)
            sim_scores.append(score / len(test_code_embedding))
            # sim_scores.append(score)
        sorted_indexes = [(i, v) for i, v in sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)]
        similar_traincase_ids = [(traincases[i]["ids"], v) for i, v in sorted_indexes[:10]]
        data = {"ids": testcases[i]["ids"], "sim_code": similar_traincase_ids}
        res += f"{json.dumps(data)}\n"

    path = f'dataset/{dataset}/test/intents/sim_token'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/sim_token_score.{intent}', 'w+') as fp:
        fp.write(res)
    return res


def preprocess_samples_semantic(intent, dataset="tlcodesum"):
    device = "cuda"
    model = SentenceTransformer("./st-codesearch-distilroberta-base").to(device)
    # model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    # https://huggingface.co/flax-sentence-embeddings/ st-codesearch-distilroberta-base

    testcases = []
    traincases = []
    res = ""
    with open(f"dataset/{dataset}/train/intents/{intent}.train", "r") as f_train, \
            open(f"dataset/{dataset}/test/intents/{intent}.test", "r") as f_test:
        testcases = [item for item in jsonlines.Reader(f_test)]
        traincases = [item for item in jsonlines.Reader(f_train)]

    test_codes_embeddings, training_codes_embeddings = [], []
    st = time.time()
    print('Test code embedding generation start...')
    for i in tqdm(range(len(testcases))):
        test_code = testcases[i]["code"]
        code1_emb = model.encode(test_code, convert_to_tensor=True)
        # code1_emb = model.encode(test_code, convert_to_tensor=True)
        test_codes_embeddings.append(code1_emb)
    ed = time.time()
    print('Test code embedding generate finish!')
    print(str(ed - st))
    print('Training code embedding generation start...')
    for i in tqdm(range(len(traincases))):
        train_code = traincases[i]["code"]
        code1_emb = model.encode(train_code, convert_to_tensor=True)
        training_codes_embeddings.append(code1_emb)
    print('Training code embedding generate finish!')

    hits = util.semantic_search(test_codes_embeddings, training_codes_embeddings, query_chunk_size=1000, corpus_chunk_size=700000, top_k=10)
    for i in tqdm(range(len(testcases))):
        similar_traincase_ids = [(traincases[item['corpus_id']]["ids"], item['score']) for item in hits[i]]
        data = {"ids": testcases[i]["ids"], "sim_code": similar_traincase_ids}
        res += f"{json.dumps(data)}\n"

    path = f'dataset/{dataset}/test/intents/sim_semantic'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/sim_semantic_score.{intent}', 'w+') as fp:
        fp.write(res)
    return res


def cal_sim_score(intent, dataset="tlcodesum"):
    with open(f'dataset/{dataset}/test/intents/sim_semantic.{intent}', 'r') as f:
        sim_test_data = [item for item in jsonlines.Reader(f)]
    with open(f"dataset/{dataset}/train/intents/{intent}.train", "r") as f_train, \
            open(f"dataset/{dataset}/test/intents/{intent}.test", "r") as f_test:
        testcases = {item['ids']: item['code'] for item in jsonlines.Reader(f_test)}
        traincases = {item['ids']: item['code'] for item in jsonlines.Reader(f_train)}
    device = "cuda"
    model = SentenceTransformer("./st-codesearch-distilroberta-base").to(device)
    res = ''
    for data in tqdm(sim_test_data):
        testcase_ids = data["ids"]
        test_code = testcases[testcase_ids]
        test_code_emb = model.encode(test_code, convert_to_tensor=True)
        tmp = {'ids': data["ids"], 'sim_code': []}
        for traincase_ids in data['sim_code_ids']:
            train_code = traincases[traincase_ids]
            train_code_emb = model.encode(train_code, convert_to_tensor=True)
            hits = util.semantic_search(test_code_emb, train_code_emb)[0]
            top_hit = hits[0]
            score = top_hit['score']
            tmp['sim_code'].append((traincase_ids, score))

        res += f"{json.dumps(tmp)}\n"

    with open(f'dataset/{dataset}/test/intents/sim_semantic_score.{intent}', 'w+') as fp:
        fp.write(res)


def preprocess_samples_quality_codet5(dataset="tlcodesum", mode="train"):
    # checkpoint = "Salesforce/codet5p-110m-embedding"
    checkpoint = "./codet5p-110m-embedding"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
    embedding = model(inputs)[0]
    print(f'Dimension of the embedding: {embedding.size()[0]}, with norm={embedding.norm().item()}')
    print(embedding)
    with open(f"dataset/{dataset}/{mode}/intents/all.{mode}", "r+", encoding="utf-8") as f_all, \
            open(f"dataset/{dataset}/{mode}/intents/quality.{mode}", "w+", encoding="utf-8") as output:
        all = [item for item in jsonlines.Reader(f_all)]
        res = {}
        for item in tqdm(all):
            code = tokenizer.encode(item["raw_code"], return_tensors="pt", truncation=True).to(device)
            comment = tokenizer.encode(item["comment"], return_tensors="pt", truncation=True).to(device)
            embedding_code = model(code)[0].cpu().detach().numpy()
            embedding_comment = model(comment)[0].cpu().detach().numpy()
            cos_sim = 1 - spatial.distance.cosine(embedding_code, embedding_comment)
            res[item["ids"]] = cos_sim

        json.dump(res, output)


def statistic_quality(dataset="tlcodesum", mode="train"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    with open(f"dataset/{dataset}/{mode}/intents/quality.{mode}", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):  # 只有一个json数据
            print(np.average(list(item.values())))
            new_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            new_y = [0 for i in new_x]
            for i in list(item.values()):
                for region_id in range(len(new_x)):
                    if i <= new_x[region_id]:
                        new_y[region_id] += 1
                        break
            plt.bar(new_x, new_y, width=0.05)
            a = sum(new_y[:5])
            b = sum(new_y)
            plt.title(f'{a} / {b} = {a / b}')
            plt.show()


def select_demo(select_num=20):
    intents = ['what', 'why', 'done', 'property', 'usage']
    with open(f"dataset/tlcodesum/test/only_in_dome_ids.test", "r", encoding="utf8") as f:
        only_in_dome_ids = [i.strip() for i in f.readlines()]
    res = {}

    for intent in intents:
        with open(f'dataset/tlcodesum/test/intents/{intent}.test', 'r') as f:
            testcases_ids = [item['ids'] for item in jsonlines.Reader(f)]
            low_testcases_ids = list(set(testcases_ids) & (set(only_in_dome_ids)))
            high_testcases_ids = list(set(testcases_ids) - set(only_in_dome_ids))

            selected_low_testcases_ids = random.sample(low_testcases_ids, select_num)
            selected_high_testcases_ids = random.sample(high_testcases_ids, select_num)

            print(selected_low_testcases_ids)
            print(selected_high_testcases_ids)

            res[intent] = {"low": selected_low_testcases_ids, "high": selected_high_testcases_ids}

    with open('dataset/tlcodesum/test/samples.test', 'w+') as f:
        json.dump(res, f)


if __name__ == '__main__':
    # import os
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # code_sentance_split('dataset/funcom/test/raw_code.test',
    #                     'dataset/funcom/test/code_split.test',
    #                     'dataset/funcom/test/code_split_format.test',
    #                     'dataset/funcom/test/code_split_format_error_ids.test')
    # category_code4intent_classifier(mode="train")
    # category_code4intents(dataset='funcom',mode='train')
    # category_code4intents(dataset='funcom',mode='test')
    # category_code4all(dataset='funcom',mode='train')
    # category_code4all(dataset='funcom',mode='valid')
    category_code4code_search_comment2code(trainingset_percentage=0.5)
    category_code4code_search_comment2code(trainingset_percentage=0.75)
    # category_code4code_search_comment2code(dataset='funcom',mode="train", neg_samples_num=10)
    # select_demo()
    # category_code4code_search_code2comment(mode="train", neg_samples_num=10)
    # intents = ["what", "why", "done", "usage", "property"]
    # dataset = "funcom"
    # res = ''
    # for intent in intents:
    #     with open(f'dataset/{dataset}/test/intents/sim_token/sim_token_score.{intent}', 'r') as f:
    #         res += f.read()
    #
    # with open(f'dataset/{dataset}/test/intents/sim_token/sim_token_score.all', 'w+') as f:
    # # with open(f'dataset/{dataset}/test/intents/sim_token/sim_token.all', 'w+') as f:
    #     f.write(res)

    # preprocess_samples_quality_codet5(dataset='funcom',mode='train')
    # statistic_quality(mode='test')
