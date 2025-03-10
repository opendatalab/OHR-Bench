# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


from typing import Callable

import evaluate
import jieba
import regex
import uuid
import numpy as np
from loguru import logger
from text2vec import Similarity
from collections import Counter

def catch_all_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # raise e
            logger.warning(repr(e))
            if func.__name__ == "bleu_score":
                return 0, 0, 0, 0, 0
            else:
                return -1
    return wrapper


@catch_all_exceptions
def bleu_score(
    continuation: str,
    reference: str,
    with_penalty = False
) -> float:
    bleu = evaluate.load('src/.cache/huggingface/bleu',
                           experiment_id=str(uuid.uuid4()))
    results = bleu.compute(predictions=[continuation], references=[[reference]])
    
    bleu_avg = results['bleu']
    bleu1 = results['precisions'][0]
    bleu2 = results['precisions'][1]
    bleu3 = results['precisions'][2]
    bleu4 = results['precisions'][3]
    brevity_penalty = results['brevity_penalty']

    if with_penalty:
        return bleu_avg, bleu1, bleu2, bleu3, bleu4
    else:
        return 0.0 if brevity_penalty==0 else bleu_avg/brevity_penalty, bleu1, bleu2, bleu3, bleu4


@catch_all_exceptions
def rougeL_score(
    continuation: str,
    reference: str
) -> float:
    rouge = evaluate.load('src/.cache/huggingface/rouge',
                           experiment_id=str(uuid.uuid4()))
    results = rouge.compute(predictions=[continuation], references=[[reference]], rouge_types=['rougeL'])
    score = results['rougeL']
    return float(score)


@catch_all_exceptions
def kw_precision(
    continuation: str,
    reference: str,
    kw_extracter: Callable[[str], list[str]],
    with_kw_list: bool = True
) -> float | tuple[float, list[str], list[str]]:
    """Measure the rationality of a generated continuation sentence with respect to the original news object."""
    kws = kw_extracter(continuation)
    if len(kws) == 0:
        return 0, [], [] if with_kw_list else 0
    appeared_kws = [kw for kw in kws if kw in reference]
    precision = len(appeared_kws) / len(kws)
    return precision, appeared_kws, kws if with_kw_list else precision


@catch_all_exceptions
def bert_score(
    continuation: str,
    reference: str
) -> float:
    """
    Note:
        Requesting the network to connect to Hugging Face. 
    """
    sim = Similarity(model_name_or_path="src/.cache/bert-large-uncased")
    score = sim.get_score(continuation, reference)
    # score = bert_score.score(continuation, reference)[0]
    return score


def classifications(
    predictions: list[bool],
    references: list[bool]
) -> tuple[float, float, float, float]:
    """
    Calculate accuracy, precision, recall, and F1 in a binary classification problem.

    Args:
        predictions (list[bool]): List of predicted values (0 or 1).
        references (list[bool]): List of true values (0 or 1).

    Returns:
        tuple: Accuracy, precision, recall, and F1 scores.

    """
    true_positive = sum(1 for a, b in zip(references, predictions) if a == 1 and b == 1)
    false_positive = sum(1 for a, b in zip(references, predictions) if a == 0 and b == 1)
    false_negative = sum(1 for a, b in zip(references, predictions) if a == 1 and b == 0)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = sum(1 for a, b in zip(references, predictions) if a == b) / len(predictions) if len(predictions) > 0 else 0
    return accuracy, precision, recall, f1


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em(prediction, ground_truth, normalize_fn):
    # print(f'In EM, normalize_fn(prediction) = {normalize_fn(prediction)} normalize_fn(ground_truth)) = {normalize_fn(ground_truth)} float(normalize_fn(prediction) in normalize_fn(ground_truth)) = {float(normalize_fn(prediction) in normalize_fn(ground_truth))}')
    return float(normalize_fn(ground_truth) in normalize_fn(prediction))


def em_strict(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def f1_en(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_zh(prediction, ground_truth, normalize_fn):
    prediction_tokens = ' '.join(jieba.cut(normalize_fn(prediction)))
    ground_truth_tokens = ' '.join(jieba.cut(normalize_fn(ground_truth)))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def has_chn_character(s):
    """chn"""
    import unicodedata
    for char in s:
        try:
            if 'CJK' in unicodedata.name(char):
                return True
        except ValueError:
            continue
    return False

@catch_all_exceptions
def exact_match_score(prediction, ground_truth):
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0


@catch_all_exceptions
def exact_match_strict_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    if isinstance(ground_truths[0], str):
        return float(max([em_strict(prediction, gt, normalize_fn) for gt in ground_truths]))
    else:
        all_scores = []
        for gts in ground_truths:
            score = np.array([em_strict(prediction, gt, normalize_fn) for gt in gts]).mean()
            all_scores.append(score)
        return float(max(all_scores))


@catch_all_exceptions
def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = 0

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    if has_chn_character(prediction) or has_chn_character(ground_truth):
        prediction_tokens = jieba.lcut(normalized_prediction)
        ground_truth_tokens = jieba.lcut(normalized_ground_truth)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


@catch_all_exceptions
def lcs_score(prediction, ground_truth):  # A is the gold standard, B is the predicted output
    A = normalize_answer(ground_truth).split()
    B = normalize_answer(prediction).split()
    dp = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
    # Fill the matrix in a bottom-up manner
    for i in range(1, len(A) + 1):
        for j in range(1, len(B) + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # backtrack and find the word-level LCS
    lcs_words = []
    i, j = len(A), len(B)
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:
            lcs_words.insert(0, A[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    if len(A) == 0:
        precision = 0.5
    else:
        precision = len(lcs_words) / len(A)
    return precision


if __name__ == "__main__":
    while True:
        _str = input()
        print(normalize_answer(_str))