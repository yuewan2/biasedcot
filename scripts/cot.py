# General dependencies
import os
import re
import copy
import time
import pickle
import argparse
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from biasedcot.data.hf_dataset import HFDataset
from biasedcot.utils.hf_llm import print_chat, extract_rationale
from biasedcot.utils.hf_llm import batch_get_answer_scores
from biasedcot.utils.hf_llm import batch_generate, batch_sample

AL = ['(a)', '(b)', '(c)', '(d)', '(e)']


def main(args):
    device = args.device
    model_option = args.model_option
    dataset_name = args.dataset_name
    splits = args.splits
    batch_size = args.batch_size
    max_length = 512
    prompt_path = '../zeroshot_cot.txt'
    stop_strings = None  # ['\n\n']

    # build model
    print(f'Building {model_option} model...')
    if model_option == 'mistral-7B':
        model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', device_map=device)
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    elif model_option == 'llama3-8B':
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map=device)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', padding_side='left')
    elif model_option == 'olmo-7B':
        model = AutoModelForCausalLM.from_pretrained('allenai/OLMo-2-1124-7B-Instruct', device_map=device)
        tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-2-1124-7B-Instruct', padding_side='left')
    else:
        raise NotImplementedError

    # build dataset
    print(f'Building {dataset_name} {splits} dataset...')
    dataset = HFDataset(dataset_name, splits)

    # build cot prompt
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f'Prompt file {prompt_path} not found.')

    with open(prompt_path, 'r') as f:
        cot_prompt = ''.join(f.readlines())
    cot_prompt += '\n\n'
    print(cot_prompt)

    instruct_prompt = "Let's think step by step. To solve the question, we need to"

    # perform cot reasoning
    output_list, batch_list = [], []
    for idx in tqdm(range(len(dataset))):
        output = {}
        source = dataset.get_data(idx)[0]
        targets = dataset.get_data(idx)[1]
        gt_idx = dataset.get_data(idx)[2]
        base_scores = batch_get_answer_scores(model, tokenizer, source, targets)
        base_entropy = entropy(softmax(base_scores), base=2)

        output['source'] = source
        output['targets'] = targets
        output['gt_idx'] = gt_idx
        output['base_scores'] = np.array(base_scores)
        output['base_idx'] = int(np.argmax(base_scores))
        output['base_entropy'] = float(base_entropy)
        batch_list.append(output)

        if len(batch_list) == batch_size:
            # batch-wise vanilla cot reasoning
            prompt_list = []
            for output in batch_list:
                prompt_list.append(cot_prompt + output['source'] + '\n' + instruct_prompt)

            if args.n_output == 1:
                contents = batch_generate(
                    model, tokenizer, prompt_list, max_length=max_length, stop_strings=stop_strings)
            elif args.n_output > 1:
                if model_option == 'llama3-8B':
                    stop_strings = ['#']
                    temperature = 0.3
                    top_p = 0.7
                else:
                    temperature = 0.9
                    top_p = 0.9
                contents = batch_sample(
                    model, tokenizer, prompt_list, max_length=max_length, stop_strings=stop_strings, n=args.n_output,
                    temperature=temperature, top_p=top_p)
            else:
                raise NotImplementedError

            contents = [instruct_prompt + ' ' + content for content in contents]
            contents = [contents[j * args.n_output: (j + 1) * args.n_output] for j in range(len(batch_list))]

            for j, sub_contents in enumerate(contents):
                output = batch_list[j]

                if args.n_output == 1:  # greedy coding with args.n_output == 1
                    content = sub_contents[0]
                    cot = re.sub('\n+', ' ', content)
                    cot = cot.replace('  ', ' ')
                    rationale, _ = extract_rationale(cot)
                    therefore_targets = [f'Therefore, t{target[1:]}' for target in output['targets']]
                    rt_scores = np.array(batch_get_answer_scores(
                        model, tokenizer, output['source'], therefore_targets, f'\nRationale: {rationale}'))

                    output['cot'] = cot
                    output['cot_f'] = rationale
                    output['cot_f_scores'] = np.array(rt_scores)
                    output['cot_f_idx'] = int(np.argmax(rt_scores))
                    output['cot_f_entropy'] = float(entropy(softmax(rt_scores), base=2))

                else:  # sampling coding with args.n_output > 1
                    output['cot'] = []
                    output['cot_f'] = []
                    output['cot_f_scores'] = []
                    output['cot_f_idx'] = []
                    output['cot_f_entropy'] = []
                    for content in sub_contents:
                        cot = re.sub('\n+', ' ', content)
                        cot = cot.replace('  ', ' ')
                        rationale, _ = extract_rationale(cot)
                        therefore_targets = [f'Therefore, t{target[1:]}' for target in output['targets']]
                        rt_scores = np.array(batch_get_answer_scores(
                            model, tokenizer, output['source'], therefore_targets, f'\nRationale: {rationale}'))

                        output['cot'].append(cot)
                        output['cot_f'].append(rationale)
                        output['cot_f_scores'].append(np.array(rt_scores))
                        output['cot_f_idx'].append(int(np.argmax(rt_scores)))
                        output['cot_f_entropy'].append(float(entropy(softmax(rt_scores), base=2)))

                output_list.append(output)

            batch_list = []

    if batch_list:
        # batch-wise vanilla cot reasoning
        prompt_list = []
        for output in batch_list:
            prompt_list.append(cot_prompt + output['source'] + '\n' + instruct_prompt)

        if args.n_output == 1:
            contents = batch_generate(
                model, tokenizer, prompt_list, max_length=max_length, stop_strings=stop_strings)
        elif args.n_output > 1:
            contents = batch_sample(
                model, tokenizer, prompt_list, max_length=max_length, stop_strings=stop_strings, n=args.n_output)
        else:
            raise NotImplementedError

        contents = [instruct_prompt + ' ' + content for content in contents]
        contents = [contents[j * args.n_output: (j + 1) * args.n_output] for j in range(len(batch_list))]

        for j, sub_contents in enumerate(contents):
            output = batch_list[j]

            if args.n_output == 1:  # greedy coding with args.n_output == 1
                content = sub_contents[0]
                cot = re.sub('\n+', ' ', content)
                cot = cot.replace('  ', ' ')
                rationale, _ = extract_rationale(cot)
                therefore_targets = [f'Therefore, t{target[1:]}' for target in output['targets']]
                rt_scores = np.array(batch_get_answer_scores(
                    model, tokenizer, output['source'], therefore_targets, f'\nRationale: {rationale}'))

                output['cot'] = cot
                output['cot_f'] = rationale
                output['cot_f_scores'] = np.array(rt_scores)
                output['cot_f_idx'] = int(np.argmax(rt_scores))
                output['cot_f_entropy'] = float(entropy(softmax(rt_scores), base=2))

            else:  # sampling coding with args.n_output > 1
                output['cot'] = []
                output['cot_f'] = []
                output['cot_f_scores'] = []
                output['cot_f_idx'] = []
                output['cot_f_entropy'] = []
                for content in sub_contents:
                    cot = re.sub('\n+', ' ', content)
                    cot = cot.replace('  ', ' ')
                    rationale, _ = extract_rationale(cot)
                    therefore_targets = [f'Therefore, t{target[1:]}' for target in output['targets']]
                    rt_scores = np.array(batch_get_answer_scores(
                        model, tokenizer, output['source'], therefore_targets, f'\nRationale: {rationale}'))

                    output['cot'].append(cot)
                    output['cot_f'].append(rationale)
                    output['cot_f_scores'].append(np.array(rt_scores))
                    output['cot_f_idx'].append(int(np.argmax(rt_scores)))
                    output['cot_f_entropy'].append(float(entropy(softmax(rt_scores), base=2)))

            output_list.append(output)

    if args.n_output == 1:
        with open(f'../results/{model_option}_{dataset_name}_{splits}.pk', 'wb') as f:
            pickle.dump(output_list, f)
    else:
        with open(f'../results/{model_option}_{dataset_name}_{splits}_sampling.pk', 'wb') as f:
            pickle.dump(output_list, f)


if __name__ == '__main__':
    # define argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_option', type=str, default='llama3-8B', choices=['mistral-7B', 'llama3-8B', 'olmo-7B'])
    parser.add_argument('--dataset_name', type=str, default='social_i_qa', choices=[
        'social_i_qa', 'commonsense_qa', 'piqa',  'logiqa', 'aqua', 'strategyqa', 'strategyqa_fact'])
    parser.add_argument('--splits', type=str, default='validation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_output', type=int, default=1)
    args = parser.parse_args()

    main(args)
