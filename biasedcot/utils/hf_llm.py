import copy
import torch
import numpy as np
import re
from typing import Tuple, List
from nltk import sent_tokenize


def extract_rationale(raw_cot):
    filtered_sents = []
    garbage_sents = []
    rationale_sents = sent_tokenize(raw_cot)
    for i, sent in enumerate(rationale_sents):
        if not re.match('Therefore, .*answer is.*', sent):
            filtered_sents.append(rationale_sents[i])
        else:
            garbage_sents.append(rationale_sents[i])
    return ' '.join(filtered_sents), ' '.join(garbage_sents)


def batch_get_answer_scores(model, tokenizer, source, targets, prompt='', verbose=False):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    if isinstance(prompt, list):
        assert len(prompt) == len(targets)
    else:
        prompt = [prompt] * len(targets)

    text_list, answer_pos_list = [], []
    for j in range(len(targets)):
        target = targets[j]
        source_prompt = source + prompt[j]

        prefix_len = tokenizer(source_prompt, return_tensors="pt").input_ids.size(1)
        answer_pos = prefix_len + 3  # currently hard-coded for Llama2 and Mistral models
        answer_pos_list.append(answer_pos)

        source_text = source_prompt + '\n' + target
        text_list.append(source_text)

    if verbose:
        for text in text_list:
            print(text)
            print('----')

    source_logprobs = batch_get_logprobs(text_list, model, tokenizer, sos_token=True)

    results = []
    for i in range(len(source_logprobs)):
        answer_logprobs = [l[1] for l in source_logprobs[i][answer_pos_list[i]:]]
        results.append(np.mean(answer_logprobs))

    return results


def batch_generate(model, tokenizer, prompt_list, max_length=512, stop_strings=None):
    inputs = tokenizer(prompt_list, padding=True, return_tensors="pt").to(model.device)

    if stop_strings is not None:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # Adjust the maximum length of the generated text
            do_sample=False,  # Use sampling for diverse outputs
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=stop_strings,
            tokenizer=tokenizer,
            top_p=None,
            temperature=None
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # Adjust the maximum length of the generated text
            do_sample=False,  # Use sampling for diverse outputs
            pad_token_id=tokenizer.eos_token_id,
            top_p=None,
            temperature=None
        )
    contents = []
    for output in outputs:
        content = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        contents.append(content.strip())
    return contents


def batch_sample(model, tokenizer, prompt_list, max_length=512,
                 stop_strings=None, n=2, temperature=0.9, top_p=0.9):
    inputs = tokenizer(prompt_list, padding=True, return_tensors="pt").to(model.device)

    if stop_strings is not None:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # Adjust the maximum length of the generated text
            do_sample=True,  # Use sampling for diverse outputs
            num_return_sequences=n,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=stop_strings,
            tokenizer=tokenizer
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # Adjust the maximum length of the generated text
            do_sample=True,  # Use sampling for diverse outputs
            num_return_sequences=n,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = outputs.view(inputs['input_ids'].size(0), n, -1)

    contents = []
    for i in range(outputs.size(0)):
        for j in range(outputs.size(1)):  # repeated n times
            output = outputs[i, j]
            content = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            contents.append(content.strip())
    return contents


def print_chat(messages):
    print('-'*40)
    for message in messages:
        if message['role'] == 'user':
            print('<|user|>')
        elif message['role'] == 'assistant':
            print('<|assistant|>')
        else:
            raise ValueError('role_name should be either "user" or role_name')
        print(message['content'])
        print('-' * 40)


def parse_raw_rationale(raw_rationale, choices, gt_idx, include_prediction=False):
    """
    Parse the raw rationale to get the rationale, prediction, and whether the prediction is correct
    :param raw_rationale: raw text (rationale) generated by the LLMs
    :param choices: a list of answer choices in the format of ['(a) choice1', '(b) choice2', ...]
    :param gt_idx: the ground truth index of the answer
    :param include_prediction: whether to include the prediction in the rationale
    :return:
    """
    choices_text = [' '.join(choice.split(' ')[1:]).lower() for choice in choices]
    choices_label = [choice.split(' ')[0] for choice in choices]

    if not raw_rationale:
        return None, None, False

    if 'Commonsense rationale:' in raw_rationale:   # llama2-7B tends to make this
        raw_rationale = raw_rationale.replace('Commonsense rationale:', '').strip()
    if 'Commonsense Rationale:' in raw_rationale:   # llama2-7B tends to make this
        raw_rationale = raw_rationale.replace('Commonsense Rationale:', '').strip()
    # if re.match('Commonsense rationale:(.*)', raw_rationale):   # llama2-7B tends to make this
    #     raw_rationale = re.match('Commonsense rationale:(.*)', raw_rationale).group(1).strip()

    rationale = raw_rationale.lower()
    rationale_sents = sent_tokenize(rationale)
    prediction = rationale_sents[-1]

    # see if one answer is mentioned in prediction
    existing_text = set()
    for text in choices_text:
        if text in prediction:
            existing_text.add(text)
    existing_text = list(existing_text)

    if len(existing_text) == 1:
        pred_idx = np.argwhere(existing_text[0] == np.array(choices_text))[0][0]
    else:
        # see if one label is mentioned in prediction
        existing_label = []
        for label in choices_label:
            # if f'({label.lower()})' in prediction:
            if label.lower() in prediction:
                existing_label.append(label)
        if len(existing_label) == 1:
            pred_idx = np.argwhere(existing_label[0] == np.array(choices_label))[0][0]
        else:
            pred_idx = -1

    if pred_idx == -1:
        return None, None, False
    else:
        rationale_sents = sent_tokenize(raw_rationale)
        prediction = rationale_sents[-1]
        if include_prediction:
            rationale = ' '.join(rationale_sents)
        else:
            rationale = ' '.join(rationale_sents[:-1])
        return rationale, prediction, pred_idx == gt_idx


def batch_get_logprobs(text_list: List[str],
                       model: torch.nn.Module,
                       tokenizer: torch.nn.Module,
                       sos_token: bool = True) -> List[Tuple[str, float]]:
    """
    Get log probabilities from hf LLM models for each token in the input text
    :param text_list: the list of input text
    :param model: the huggingface model
    :param tokenizer: the huggingface tokenizer
    :param sos_token: whether the llms has <sos> token
    :return: a list of tuples, each containing a token and its log probability
    """
    model.eval()

    # Get the device of the model
    device = next(model.parameters()).device

    # Pass the text to the models
    input_ids = tokenizer(text_list, padding=True, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones(input_ids.size()).long().to(device)
    attention_mask[input_ids == 2] = 0
    padding_nums = (input_ids == 2).sum(-1)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)

    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    token_log_prob_list = []
    for i in range(log_probs.size(0)):
        padding = 0

        # Align the input_ids and the log_probs
        if sos_token:
            shifted_input_ids = input_ids[i, 1 + padding_nums[i]:]
            shifted_log_probs = log_probs[i, padding_nums[i]:-1]
            padding += 1
        else:  # So far only for Falcon model
            shifted_input_ids = input_ids[i]
            shifted_log_probs = log_probs[i]

        log_probs_for_tokens = shifted_log_probs.gather(1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        token_list = tokenizer.convert_ids_to_tokens(input_ids[i][padding + padding_nums[i]:].squeeze())
        log_probs_list = log_probs_for_tokens.squeeze().tolist()
        token_log_prob = list(zip(token_list, log_probs_list))
        token_log_prob_list.append(token_log_prob)

    return token_log_prob_list


def get_logprobs(text: str,
                 model: torch.nn.Module,
                 tokenizer: torch.nn.Module,
                 sos_token: bool = True) -> List[Tuple[str, float]]:
    """
    Get log probabilities from hf LLM models for each token in the input text
    :param text: the input text
    :param model: the huggingface model
    :param tokenizer: the huggingface tokenizer
    :param sos_token: whether the llms has <sos> token
    :return: a list of tuples, each containing a token and its log probability
    """
    model.eval()
    padding = 0

    # Get the device of the model
    device = next(model.parameters()).device

    # Pass the text to the models
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    # Align the input_ids and the log_probs
    if sos_token:
        shifted_input_ids = input_ids[:, 1:]
        shifted_log_probs = log_probs[:, :-1]
        padding += 1
    else:  # So far only for Falcon model
        shifted_input_ids = input_ids
        shifted_log_probs = log_probs

    log_probs_for_tokens = shifted_log_probs.gather(2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    # Convert log probabilities to readable format
    token_list = tokenizer.convert_ids_to_tokens(input_ids.squeeze())[padding:]
    log_probs_list = log_probs_for_tokens.squeeze().tolist()

    return list(zip(token_list, log_probs_list))


def parse_scoring_results(results, selected_inds: List[int] = None, offset: int = 0):
    """
    Parse the scoring results computed by the LLMs from `llms_scoring.ipynb`
    :param results: list of tuple(list, int) that contains the logprobs for each token and the answer position
    :param selected_inds: list of indices that are selected for the evaluation
    :param offset: the offset for the answer position
    :return: a dictionary of the output that includes prompt perplexity, the raw logprobs, and the prediction
    """
    prompt_perplexities, raw_scores, predicts = [], [], []
    for i in range(len(results)):
        if selected_inds is not None and i not in selected_inds:
            continue

        logprobs_list, answer_pos = results[i]
        prompt_logprob_with_token = logprobs_list[0][:answer_pos+offset]  # prompt should be the same for each answer choice
        prompt_logprob = np.mean([p[1] for p in prompt_logprob_with_token])
        prompt_perplexity = np.exp(-prompt_logprob)

        answer_logprobs_with_token = [logprobs[answer_pos+offset:] for logprobs in logprobs_list]
        answer_logprobs = [np.mean([a[1] for a in alwt]) for alwt in answer_logprobs_with_token]

        prompt_perplexities.append(prompt_perplexity)
        raw_scores.append(answer_logprobs)
        predicts.append(np.argmax(answer_logprobs))
    return {'prompt_perplexity': prompt_perplexities, 'raw_score': raw_scores, 'predict': predicts}


def parse_generative_results(results, target_choices, selected_inds: List[int] = None):
    """
    Parse the generated results computed by the LLMs from `llms_generative.ipynb`
    :param results:
    :param target_choices: list of list that contains all answers choices in the text form
    :param selected_inds: list of indices that are selected for the evaluation
    :return: a dictionary of the output that includes prompt perplexity, the raw logprobs, and the prediction
    """
    assert len(results) == len(target_choices)

    def parse_generation(generation, target_choices):
        # Filter #1: directly looking for answer choice exact match
        pred_choice = '<empty>'
        for j, target_choice in enumerate(target_choices):
            target_choice = target_choice.replace('(', '\(')
            target_choice = target_choice.replace(')', '\)')
            target_choice = f'.*({target_choice}).*'
            if re.match(target_choice, generation):
                pred_choice = target_choices[j]
                break

        # Filter #2: looking for the selection of option e.g., (a). The exact wording may not match.
        if pred_choice == '<empty>':
            label2idx = {'(a)': 0, '(b)': 1, '(c)': 2, '(d)': 3, '(e)': 4}
            label = re.match('.*(\([a-z]\)).*', generation)
            if label is not None and label.group(1) in label2idx:
                label = label.group(1)
                pred_choice = target_choices[label2idx[label]]
            elif label is not None and label.group(1) not in label2idx:
                print(f'[warning] missing prediction for {generation}')

        # Filter #3: looking for the exact word match in answer choice
        if pred_choice == '<empty>':
            for target_choice in target_choices:
                if target_choice[4:] in generation:
                    pred_choice = target_choice
                    break

        if pred_choice == '<empty>':
            pred_choice = generation

        return {'raw': generation, 'predict_choice': pred_choice}

    predicts_a, predicts_b = [], []
    raw_predicts_a, raw_predicts_b = [], []
    for i, outcomes in enumerate(results):
        if selected_inds is not None and i not in selected_inds:
            continue

        if isinstance(outcomes, str) == 1:
            outcome_a = outcome_b = outcomes
        else:
            outcome_a, outcome_b = outcomes

        output = parse_generation(outcome_a, target_choices[i])
        if output['predict_choice'] in target_choices[i]:
            predict = np.argwhere(np.array(target_choices[i]) == output['predict_choice']).item()
        else:
            predict = -1
        predicts_a.append(predict)
        raw_predicts_a.append(output['raw'])

        output = parse_generation(outcome_b, target_choices[i])
        if output['predict_choice'] in target_choices[i]:
            predict = np.argwhere(np.array(target_choices[i]) == output['predict_choice']).item()
        else:
            predict = -1
        predicts_b.append(predict)
        raw_predicts_b.append(output['raw'])

    return {'predict_a': predicts_a, 'raw_a': raw_predicts_a, 'predict_b': predicts_b, 'raw_b': raw_predicts_b}