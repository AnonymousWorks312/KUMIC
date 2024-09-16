import argparse
import json
import logging
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from utils4attention import processors, output_modes, logger, convert_examples_to_features, compute_metrics, convert_examples_to_features_code2comment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}


def heatmap(attention_weights, x, y, title):
    # attention_weights_head = attention_weights
    transfer = MinMaxScaler(feature_range=(0, 1))
    # transfer = StandardScaler()
    attention_weights_head = transfer.fit_transform(attention_weights)
    print(attention_weights_head)
    print(attention_weights)
    # 使用Seaborn设置热图样式
    # style.use('ggplot')
    sns.set()
    # sns.set_style('whitegrid')

    # 绘制热图
    plt.figure(figsize=(40, 8))
    sns.heatmap(attention_weights_head,
                cmap="Greens",
                annot=False,
                fmt=".2f",
                xticklabels=x,
                yticklabels=y)
    plt.title(title)

    plt.xlabel("Comment")
    plt.ylabel("Code")
    # plt.savefig('./{}.jpg'.format(title))
    plt.show()


def multi_heatmap(attention_weights_list, x, y, title):
    # 使用Seaborn设置热图样式
    # style.use('ggplot')
    sns.set()
    attention_weight_num = len(attention_weights_list)

    f, axs = plt.subplots(figsize=(40, 8 * attention_weight_num), nrows=attention_weight_num)
    for i in range(attention_weight_num):
        if (i + 1) % 2 == 0:
            sns.heatmap(attention_weights_list[i], linewidths=0.05, ax=axs[i], cmap="Greens", xticklabels=x, yticklabels=y)
        else:
            sns.heatmap(attention_weights_list[i], linewidths=0.05, ax=axs[i], cmap="Greens", yticklabels=y)
            axs[i].set_xticklabels([])  # 设置x轴图例为空值
        axs[i].set_title(title + str(i))
        axs[i].set_xlabel('')
        axs[i].set_ylabel('Code')

    # matplotlib colormap
    # sns.heatmap(attention_weights_list[-1], linewidths=0.05, ax=axs[-1], cmap="Greens", xticklabels=x, yticklabels=y)
    # # rainbow为 matplotlib 的colormap名称
    # axs[-1].set_title(title)
    # plt.xlabel("Comment")
    f.show()


def find_important_token_line(attention_weights, code_tokenid2lineid):
    '''
    :param attention_weights:多个decoder的attention_weight的列表
    :param code_tokenid2lineid: 每个code中token对应的行号
    '''
    attention_lineid2token_index = {}  # 存储每行包含哪些attention token的id
    if len(code_tokenid2lineid) == 0:
        return {}, []
    line_id = 0  # 记录当前行号
    current_line_token_num = code_tokenid2lineid[line_id]  # 当前所在行token数范围（最大值）


    # 针对生成的comment中每个token
    for i in range(len(attention_weights)):
        attention_token_index = np.where(attention_weights[i] != 0)[0]  # code中attention非零的token的在code中的index
        attention_token2lineid = []  # 存储每个attention token所在的code行

        for token_index in attention_token_index:
            while token_index >= current_line_token_num and line_id < len(code_tokenid2lineid)-1:
                line_id += 1
                current_line_token_num += code_tokenid2lineid[line_id]

            attention_token2lineid.append(line_id)

            if not attention_lineid2token_index.__contains__(line_id):
                attention_lineid2token_index[line_id] = set()
            attention_lineid2token_index[line_id].add(token_index)

    data = []

    # 计算每行code的总的attention
    # key = 行号；value = 多个decoder的attention weight和
    line_attention_weight = {line_id: float(torch.sum(attention_weights[:, list(attention_lineid2token_index[line_id])])) for line_id in attention_lineid2token_index.keys()}
    sorted_line_attention_weight = sorted(line_attention_weight.items(), key=lambda x: x[1], reverse=True)  # 根据attention由大到小排序

    return line_attention_weight, sorted_line_attention_weight


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, task, tokenizer, ttype='train', is_code2comment=False):
    # Load data features from cache or dataset file
    file_name = args.train_file.split('.')[0]
    # cached_features_file = os.path.join(args.data_dir,
    #                                     'cached_train_tlcodesum-0-spl-code-code2comment_pretrained_codebert_500_codesearch')
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    # if os.path.exists(cached_features_file):
    try:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    except:
        print("Creating features from dataset file at %s", args.data_dir)
        features = create_examples(args, task, tokenizer, ttype, is_code2comment)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_code_tokenid2lineids = torch.tensor(two_dimensional_list2tensor([f.code_tokenid2lineid for f in features]))

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_code_tokenid2lineids)
    # dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset


def create_examples(args, task, tokenizer, ttype='train', is_code2comment=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    file_name = args.train_file.split('.')[0]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    label_list = processor.get_labels()
    examples = processor.get_train_examples(args.data_dir, args.train_file)
    features = None
    if is_code2comment:
        features = convert_examples_to_features_code2comment(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                             cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                             # xlnet has a cls token at the end
                                                             cls_token=tokenizer.cls_token,
                                                             sep_token=tokenizer.sep_token,
                                                             cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                             pad_on_left=bool(args.model_type in ['xlnet']),
                                                             # pad on the left for xlnet
                                                             pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                             max_token_inline=25, max_line_num=19, max_comment_len=30)
    else:
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                max_token_inline=25, max_line_num=19, max_comment_len=30)

    print("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)

    return features


def two_dimensional_list2tensor(two_dimensional_list):
    max_len = max([len(i) for i in two_dimensional_list])
    new_two_dimensional_list = []
    for one_dimensional_list in two_dimensional_list:
        new_two_dimensional_list.append(one_dimensional_list + [0] * (max_len - len(one_dimensional_list)))
    return new_two_dimensional_list


def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode='dev', is_code2comment=False):
    results = {}
    eval_dataset = load_and_cache_examples(args, "codesearch", tokenizer, ttype='train', is_code2comment=is_code2comment)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    results = {}

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    preds = None
    res = ""
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            code_tokenid2lineids = batch[4]

            outputs = model(**inputs)
            tmp_eval_loss, logits, attentions = outputs[:3]
            # tmp_eval_loss, logits = outputs[:2]

            for i in range(len(code_tokenid2lineids)):
                token_num = sum(inputs['attention_mask'][i])
                comment_intent_code_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])[:token_num]
                attention = attentions[-1].cpu()[i][-1][:token_num, :token_num]

                sep_token_index = comment_intent_code_tokens.index(tokenizer.sep_token)
                # comment_tokens = comment_intent_code_tokens[1:sep_token_index]
                # intent_tokens = comment_intent_code_tokens[sep_token_index + 1:sep_token_index + 2]
                # code_tokens = comment_intent_code_tokens[sep_token_index + 2:-1]

                # attentions[-1][i][-1][1:sep_token_index]
                if is_code2comment:
                    line_attention_weight, sorted_line_attention_weight = find_important_token_line(attention_weights=attention[sep_token_index + 1:-1, 2:sep_token_index],
                                                                                                    code_tokenid2lineid=[int(num) for num in code_tokenid2lineids[i].cpu() if num > 0])
                else:
                    line_attention_weight, sorted_line_attention_weight = find_important_token_line(attention_weights=attention[1:sep_token_index, sep_token_index + 2:-1],
                                                                                                    code_tokenid2lineid=[int(num) for num in code_tokenid2lineids[i].cpu() if num > 0])
                # heatmap(attention, x=comment_intent_code_tokens, y=comment_intent_code_tokens,
                #         title="No.{}, intent = {}, Attention head = {}".format(-1, intent_tokens, -1))
                # heatmap(attention[1:sep_token_index, sep_token_index + 2:-1], x=code_tokens, y=comment_tokens,
                #         title="No.{}, intent = {}, Attention head = {}".format(pos, intent_tokens, -1))
                # print(f"attention[1:sep_token_index,1:sep_token_index] = {torch.sum(attention[1:sep_token_index, 1:sep_token_index], dim=1)}")
                # print(f"attention[1:sep_token_index,sep_token_index+1:-1] = {torch.sum(attention[1:sep_token_index, sep_token_index + 2:-1], dim=1)}")
                # print(f"attention[sep_token_index+1:-1,1:sep_token_index] = {torch.sum(attention[sep_token_index + 2:-1, 1:sep_token_index], dim=1)}")
                # print(f"attention[sep_token_index+1:-1,sep_token_index+1:-1] = {torch.sum(attention[sep_token_index + 2:-1, sep_token_index + 2:-1], dim=1)}")
                # print(f"======={pos}======")

                res += (json.dumps(line_attention_weight) + '\n')
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    path = f'result/{args.dataset}'
    os.makedirs(path, exist_ok=True)
    with open(rf'{path}/code_stat_attention_weight-20_comment2code_0.75.train', 'w+') as f:
        f.write(res)
    preds_label = np.argmax(preds, axis=1)
    result = compute_metrics("codesearch", preds_label, out_label_ids)
    print("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        print("  %s = %s", key, str(result[key]))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dataset = "tlcodesum"

    parser.add_argument("--dataset", default=f"tlcodesum", type=str,
                        help="Task dataset")
    parser.add_argument("--data_dir", default=f"dataset/code_search/{dataset}", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="/root/lishuochuan/DOME/src/pretrained_codebert", type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=f"saved_model/{dataset}", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters

    parser.add_argument("--max_seq_length", default=500, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--per_gpu_train_batch_size", default=64,  type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default=True, action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=444,
                        help="random seed for initialization")

    parser.add_argument("--train_file", default=f"{dataset}-0-spl-code-comment2code.train", type=str,
                        help="train file")
    parser.add_argument("--pred_model_dir", default="saved_model/tlcodesum-comment2code-500-0.75/checkpoint-best", type=str,
                        help='model for prediction')
    args = parser.parse_args()
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s  (%(filename)s:%(lineno)d, %(funcName)s())',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    # Setup CUDA, GPU & distributed training

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.device = device

    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    print("evaluate model which from {}".format(args.pred_model_dir))

    print('testing')
    model = model_class.from_pretrained(args.pred_model_dir, output_attentions=True)
    model.to(args.device)

    evaluate(args, model, tokenizer, checkpoint=None, prefix='', mode='train')
