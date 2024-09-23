import nltk
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
import jieba

# from rouge import Rouge

import scipy
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import sys, math, re, xml.sax.saxutils
import subprocess
import os
import argparse
import json

from eval.bleu import compute_bleu
from eval.rouge import Rouge
from eval.meteor import Meteor


# nltk.download('wordnet')


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h  # m-h, m+h


# Added to bypass NIST-style pre-processing of hyp and ref files -- wade
nonorm = 0

preserve_case = False
eff_ref_len = "shortest"

normalize1 = [
    ('<skipped>', ''),  # strip "skipped" tags
    (r'-\n', ''),  # strip end-of-line hyphenation and join lines
    (r'\n', ' '),  # join lines
    #    (r'(\d)\s+(?=\d)', r'\1'), # join digits
]
normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

normalize2 = [
    (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 '),  # tokenize punctuation. apostrophe is missing
    (r'([^0-9])([\.,])', r'\1 \2 '),  # tokenize period and comma unless preceded by a digit
    (r'([\.,])([^0-9])', r' \1 \2'),  # tokenize period and comma unless followed by a digit
    (r'([0-9])(-)', r'\1 \2 ')  # tokenize dash when preceded by a digit
]
normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]


def normalize(s):
    '''Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl.'''
    # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
    if (nonorm):
        return s.split()
    if type(s) is not str:
        s = " ".join(s)
    # language-independent part:
    for (pattern, replace) in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})
    # language-dependent part (assuming Western languages):
    s = " %s " % s
    if not preserve_case:
        s = s.lower()  # this might not be identical to the original
    for (pattern, replace) in normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()


def count_ngrams(words, n=4):
    counts = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def cook_refs(refs, n=4):
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''

    refs = [normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return ([len(ref) for ref in refs], maxcounts)


def cook_test(test, item, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''
    (reflens, refmaxcounts) = item
    test = normalize(test)
    result = {}
    result["testlen"] = len(test)

    # Calculate effective reference sentence length.

    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens)) / len(reflens)
    elif eff_ref_len == "closest":
        min_diff = None
        for reflen in reflens:
            if min_diff is None or abs(reflen - len(test)) < min_diff:
                min_diff = abs(reflen - len(test))
                result['reflen'] = reflen

    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]

    result['correct'] = [0] * n
    counts = count_ngrams(test, n)
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result


def score_cooked(allcomps, n=4, ground=0, smooth=1):
    totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}
    for comps in allcomps:
        for key in ['testlen', 'reflen']:
            totalcomps[key] += comps[key]
        for key in ['guess', 'correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
    logbleu = 0.0
    all_bleus = []
    for k in range(n):
        correct = totalcomps['correct'][k]
        guess = totalcomps['guess'][k]
        addsmooth = 0
        if smooth == 1 and k > 0:
            addsmooth = 1
        logbleu += math.log(correct + addsmooth + sys.float_info.min) - math.log(guess + addsmooth + sys.float_info.min)
        if guess == 0:
            all_bleus.append(-10000000)
        else:
            all_bleus.append(math.log(correct + sys.float_info.min) - math.log(guess))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)

    brevPenalty = min(0, 1 - float(totalcomps['reflen'] + 1) / (totalcomps['testlen'] + 1))
    for i in range(len(all_bleus)):
        if i == 0:
            all_bleus[i] += brevPenalty
        all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus


def bleu(refs, candidate, ground=0, smooth=1):
    refs = cook_refs(refs)
    test = cook_test(candidate, refs)
    return score_cooked([test], ground=ground, smooth=smooth)


def bleu_final(refs, candidate):
    score = [0] * 5
    num = 0.0
    count = 0

    bl = bleu(refs, candidate)
    score = [score[i] + bl[i] for i in range(0, len(bl))]
    num += 1

    return [s * 100.0 / num for s in score]


def individual_bleu(target, inference):
    target_fenci = ' '.join(jieba.cut(target))
    inference_fenci = ' '.join(jieba.cut(inference))


    # reference = [['this', 'is', 'a', 'duck']]
    reference = []  
    candidate = [] 
    reference.append(target_fenci.split())
    candidate = (inference_fenci.split())
    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram


def cumulative_bleu(reference, candidate):
    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram


def calculate_meteor(target, inference):

    target_fenci = ' '.join(jieba.cut(target))
    inference_fenci = ' '.join(jieba.cut(inference))


    # reference = [['this', 'is', 'a', 'duck']]
    reference = []  
    candidate = [] 

    reference.append(target_fenci.split())
    candidate = (inference_fenci.split())


    meteor = meteor_score.meteor_score(reference, candidate)



    return meteor


def calculate_rough(target, inference):

    rouge = Rouge()
    scores = rouge.get_scores(inference, target)


    return scores[0]["rouge-1"]


def calculate_all(target, inference):
    bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram = individual_bleu(target, inference)
    meteor = calculate_meteor(target, inference)
    rough_score = calculate_rough(target, inference)
    # print('bleu-cn: {}'.format(bleu_final([target], inference)[0]))
    # print('bleu 1-gram: %f' % bleu_1_gram)
    # print('bleu 2-gram: %f' % bleu_2_gram)
    # print('bleu 3-gram: %f' % bleu_3_gram)
    # print('bleu 4-gram: %f' % bleu_4_gram)
    # print("The METEOR score is:", meteor)
    # print("ROUGE-1 precision:", rough_score["p"])
    # print("ROUGE-1 recall:", rough_score["r"])
    # print("ROUGE-1 F1 score:", rough_score["f"])
    return bleu_1_gram, meteor, rough_score['f']


def calculate_all4list(target, inferences):
    bleu_1_grams = []
    meteors = []
    rough_scores = []
    for inference in inferences:
        bleu_1_grams.append(individual_bleu(target, inference)[0])
        meteors.append(calculate_meteor(target, inference))
        rough_scores.append(calculate_rough(target, inference)['f'])
    # print('bleu-cn: {}'.format(bleu_final([target], inference)[0]))
    # print('bleu 1-gram: %f' % bleu_1_gram)
    # print('bleu 2-gram: %f' % bleu_2_gram)
    # print('bleu 3-gram: %f' % bleu_3_gram)
    # print('bleu 4-gram: %f' % bleu_4_gram)
    # print("The METEOR score is:", meteor)
    # print("ROUGE-1 precision:", rough_score["p"])
    # print("ROUGE-1 recall:", rough_score["r"])
    # print("ROUGE-1 F1 score:", rough_score["f"])
    return bleu_1_grams, meteors, rough_scores


def calculate_all4list_new(target, inferences):
    bleus = []
    # meteors = []
    roughs = []
    rouge_calculator = Rouge()
    for inference in inferences:
        bleu_score = compute_bleu([[target.split()]], [inference.split()], smooth=True)[0]
        rouge_score = rouge_calculator.calc_score(inference.split(), [target.split()])
        bleus.append(bleu_score)
        roughs.append(rouge_score)
    # meteor_calculator = Meteor()
    # meteor_score, meteors = meteor_calculator.compute_score([i for i in range(len(inferences))], [inference.split() for inference in inferences], [[target.split()] for i in range(len(inferences))])

    return bleus, roughs


def cal_meteor_new(comment_pred, comment_ref):
    meteor_calculator = Meteor()
    _, meteors = meteor_calculator.compute_score([i for i in range(len(comment_pred))],
                                                 [i.split() for i in comment_pred],
                                                 [[i.split()] for i in comment_ref])

    return meteors
