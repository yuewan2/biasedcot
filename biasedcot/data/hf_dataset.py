import os
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from datasets import load_dataset
from pathlib import Path

BASE = Path(os.path.dirname(os.path.realpath(__file__)))


class HFDataset(object):
    """
    Wrapper class for Hugging Face datasets.
    """

    def __init__(self,
                 dataset_name: str,
                 split: str = 'train'):

        assert dataset_name in ['commonsense_qa', 'social_i_qa', 'piqa', 'logiqa', 'strategyqa', 'aqua', 'strategyqa_fact']
        assert split in ['train', 'validation', 'test']

        self.dataset_name = dataset_name
        self.split = split
        self.rationale_type = ''

        if dataset_name == 'commonsense_qa':
            self.dataset = load_commonsense_qa(split=split)
        elif dataset_name == 'social_i_qa':
            self.dataset = load_social_i_qa(split=split)
        elif dataset_name == 'piqa':
            self.dataset = load_piqa(split=split)
        elif dataset_name == 'logiqa':
            self.dataset = load_logiqa(split=split)
        elif dataset_name == 'strategyqa' or dataset_name == 'strategyqa_fact':
            self.dataset = load_strategyqa(split=split, factual=dataset_name == 'strategyqa_fact')
        elif dataset_name == 'aqua':
            self.dataset = load_aqua(split=split)
        else:
            raise ValueError('Invalid dataset name')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def get_data(self, i, rationale_only=False):
        if rationale_only:
            source = "Question: N/A.\nContext: N/A.\n" + self.dataset[i]['_source'].split('\n')[-1]
        else:
            source = self.dataset[i]['_source']
        targets = self.dataset[i]['_targets']
        targets = [re.sub('\n+', ' ', target) for target in targets]  # [01/24] clean-up the answer choices
        gt_idx = self.dataset[i]['_gt_index']

        if self.rationale_type:
            source += f"\nRationale: {self.dataset[i]['_rationale']}"

        return source, targets, gt_idx


def load_commonsense_qa(split: str = 'validation'):
    labels = ['A', 'B', 'C', 'D', 'E']
    hf_dataset = load_dataset('commonsense_qa', split=split)

    annot_list = []

    for i in range(len(hf_dataset)):
        annot = hf_dataset[i]
        source = 'Question: {}\nAnswer choices: '.format(
            hf_dataset[i]['question'])
        gt_index = np.argwhere(annot['answerKey'] == np.array(annot['choices']['label'])).item()

        answer_choices = ' '.join(
            ['({}) {}'.format(labels[k].lower(), annot['choices']['text'][j]) for k, j in enumerate(range(5))])
        source += answer_choices

        targets = [
            'The answer is ({}) {}.'.format(labels[k].lower(), annot['choices']['text'][j]) for k, j in
            enumerate(range(5))
        ]

        annot.update({'_source': source, '_targets': targets, '_gt_index': gt_index})
        annot_list.append(annot)

    return annot_list


def load_social_i_qa(split: str = 'validation'):
    hf_dataset = load_dataset('social_i_qa', split=split)

    annot_list = []

    for i in range(len(hf_dataset)):
        annot = hf_dataset[i]
        source = 'Question: {}\nContext: {}\nAnswer choices: '.format(
            hf_dataset[i]['question'], hf_dataset[i]['context'])
        answer_choices = [
            f"(a) {annot['answerA']}",
            f"(b) {annot['answerB']}",
            f"(c) {annot['answerC']}"
        ]
        source += ' '.join(answer_choices)  # + '.'

        gt_index = int(annot['label']) - 1

        targets = [
            f'The answer is {answer}.' for answer in answer_choices
        ]

        annot.update({'_source': source, '_targets': targets, '_gt_index': gt_index})
        annot_list.append(annot)

    return annot_list


def load_piqa(split: str = 'validation'):
    hf_dataset = load_dataset('piqa', split=split)

    annot_list = []

    for i in range(len(hf_dataset)):
        annot = hf_dataset[i]
        source = 'Question: {}\nAnswer choices: '.format(hf_dataset[i]['goal'])
        answer_choices = [
            f"(a) {annot['sol1']}",
            f"(b) {annot['sol2']}"
        ]
        source += ' '.join(answer_choices)

        gt_index = int(annot['label'])

        targets = [
            f'The answer is {answer}.' for answer in answer_choices
        ]

        annot.update({'_source': source, '_targets': targets, '_gt_index': gt_index})
        annot_list.append(annot)

    return annot_list


def load_logiqa(split: str = 'validation'):
    hf_dataset = load_dataset('EleutherAI/logiqa', split=split)

    annot_list = []
    for i in range(len(hf_dataset)):
        annot = hf_dataset[i]
        source = 'Question: {}\nContext: {}\nAnswer choices: '.format(
            hf_dataset[i]['question'], hf_dataset[i]['context'])
        answer_choices = [
            f"(a) {annot['options'][0]}",
            f"(b) {annot['options'][1]}",
            f"(c) {annot['options'][2]}",
            f"(d) {annot['options'][3]}",
        ]
        source += '\n' + '\n'.join(answer_choices)

        gt_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}[annot['label']]

        targets = [
            f'The answer is {answer}.' for answer in answer_choices
        ]

        annot.update({'_source': source, '_targets': targets, '_gt_index': gt_index})
        annot_list.append(annot)

    return annot_list


def load_strategyqa(split: str = 'validation', factual: bool = False):
    if split == 'validation':
        with open('../data/strategyqa/dev.json', 'r') as f:
            dataset = json.load(f)
    elif split == 'train':
        with open('../data/strategyqa/train.json', 'r') as f:
            dataset = json.load(f)
    else:
        raise ValueError('Invalid split')

    annot_list = []
    for i in range(len(dataset)):
        annot = dataset[i]
        if factual:
            facts = []
            for fact in annot['facts']:
                if fact[-1] == '.':
                    facts.append(fact[:-1])
                else:
                    facts.append(fact)
            source = 'Question: {}\nFacts: {}\nAnswer choices: '.format(
                dataset[i]['question'], '. '.join(facts) + '.')
        else:
            source = 'Question: {}\nAnswer choices: '.format(
                dataset[i]['question'])
        answer_choices = [
            f"(a) True",
            f"(b) False",
        ]
        source += ' '.join(answer_choices)

        if annot['answer']:
            gt_index = 0
        else:
            gt_index = 1

        targets = [
            f'The answer is {answer}.' for answer in answer_choices
        ]

        annot.update({'_source': source, '_targets': targets, '_gt_index': gt_index})
        annot_list.append(annot)
    return annot_list


def load_aqua(split: str = 'validation'):
    AL = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    if split == 'validation':
        data_path = '../data/aqua/dev.json'
    elif split == 'train':
        data_path = '../data/aqua/train.json'
    else:
        raise ValueError('Invalid split')

    dataset = []
    with open(data_path, 'r') as file:
        for line in file:
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")

    annot_list = []
    for i in range(len(dataset)):
        annot = dataset[i]
        source = 'Question: {}\nAnswer choices: '.format(
            dataset[i]['question'])

        answer_choices_text = [
            re.match('[A-Z]\)(.*)', opt).group(1) for opt in dataset[i]['options']
        ]
        answer_choices = [f'{AL[i]} {opt}' for i, opt in enumerate(answer_choices_text)]
        source += ' '.join(answer_choices)

        gt_index = mapping[dataset[i]['correct']]

        targets = [
            f'The answer is {answer}.' for answer in answer_choices
        ]

        annot.update({'_source': source, '_targets': targets, '_gt_index': gt_index})
        annot_list.append(annot)
    return annot_list
