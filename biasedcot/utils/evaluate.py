import numpy as np
import re
from collections import Counter
from itertools import product
import torch
from nltk.tokenize import sent_tokenize


def get_difficulty(scores, gt_idx):
    baseline_pred_idx = np.argmax(scores)
    if baseline_pred_idx != gt_idx:
        return scores[baseline_pred_idx] - scores[gt_idx]
    else:
        return sorted(scores)[::-1][1] - scores[baseline_pred_idx]


def get_choices(targets, include_letter=False):
    if include_letter:
        choices = [re.match('The answer is (\([a-z]\) .*)\.', target).group(1) for target in targets]
    else:
        choices = [re.match('The answer is \([a-z]\) (.*)\.', target).group(1) for target in targets]
    return choices


def get_rationale_coherent_matrix(model, tokenizer, rationale, choices):
    rationale_tokenized = sent_tokenize(rationale)

    mention_choices = [f'the sentence is talking about "{choice}".' for choice in choices]
    mention_pairs = list(product(rationale_tokenized, mention_choices))
    inputs = tokenizer(mention_pairs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs.to('cuda:0')).logits, dim=1)
    rationale_coherent_probs = probabilities[:, 2].reshape(len(rationale_tokenized), len(mention_choices)).cpu().numpy()

    return rationale_coherent_probs


def get_rationale_explicit_matrics(model, tokenizer, rationale, choices):
    rationale_tokenized = sent_tokenize(rationale)

    # positive explicitness
    explicit_choices = [f'the answer is "{choice.lower()}".' for choice in choices]
    explicit_pairs = list(product(rationale_tokenized, explicit_choices))
    inputs = tokenizer(explicit_pairs, return_tensors="pt", padding=True, truncation=True)
    device = next(model.parameters()).device
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs.to('cuda:0')).logits, dim=1)
    rationale_explicit_probs = probabilities[:, 2].reshape(len(rationale_tokenized),
                                                           len(explicit_choices)).cpu().numpy()

    # negative explicitness
    explicit_choices = [f'the answer is not "{choice.lower()}".' for choice in choices]
    explicit_pairs = list(product(rationale_tokenized, explicit_choices))
    inputs = tokenizer(explicit_pairs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs.to('cuda:0')).logits, dim=1)
    rationale_explicit_probs_neg = probabilities[:, 2].reshape(len(rationale_tokenized),
                                                               len(explicit_choices)).cpu().numpy()

    return rationale_explicit_probs, rationale_explicit_probs_neg


def majority_vote(nums):
    count = Counter(nums)
    max_count = max(count.values())  # Get the highest frequency
    candidates = [num for num, freq in count.items() if freq == max_count]
    return candidates[0]


def get_logsumexp(logits):
    # Subtract the max logit for numerical stability
    max_logit = np.max(logits)
    logits_stable = logits - max_logit
    # Compute LSE
    lse = max_logit + np.log(np.sum(np.exp(logits_stable)))
    return lse


def parse_llm_label(llm_result_list, cot_result_list):
    LM = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    llm_labels = []
    n_invalid = 0
    for i in range(len(llm_result_list)):
        sub_labels = []
        choices = get_choices(cot_result_list[i]['targets'], include_letter=False)
        for j in range(len(llm_result_list[i])):
            contents = llm_result_list[i][j]
            if not isinstance(contents, list):
                contents = [contents]

            tmp = []
            for content in contents:
                content = content.split('.')[0]
                pred_idx = -1
                if re.match('Therefore, the answer is (.*).', content):
                    pred_content = re.match('Therefore, the answer is (.*)', content).group(1)
                    pred_content = pred_content.lower()

                    if re.match('\(([a-z])\)', pred_content):
                        pred_idx = LM.get(re.match('\(([a-z])\)', pred_content).group(1), -1)
                    elif re.match('^\(?([a-z])\)?$', pred_content):
                        pred_idx = LM.get(re.match('^\(?([a-z])\)?$', pred_content).group(1), -1)
                    elif re.match('^\(?([a-z])\)? .*$', pred_content):
                        pred_idx = LM.get(re.match('^\(?([a-z])\)? .*$', pred_content).group(1), -1)
                    elif pred_content in choices:
                        pred_idx = np.argwhere(np.array(choices) == pred_content).flatten()[0]
                    elif re.match('.*\(([a-z])\)$', pred_content):
                        pred_idx = LM.get(re.match('.*\(([a-z])\)$', pred_content).group(1), -1)
                    else:
                        n_invalid += 1

                        print(f'[WARNING] Ignore invalid content of {[pred_content[:min(len(pred_content), 64)]]}')

                if pred_idx > len(choices) - 1:  # invalid predicted index
                    pred_idx = -1

                tmp.append(pred_idx)
            sub_labels.append(majority_vote(tmp))
        llm_labels.append(sub_labels)
    print(f'[SUMMARY] Found and ignore {n_invalid} number of invalid contents.')
    return llm_labels
