import os
import pickle
import argparse
from openai import OpenAI
from tqdm import tqdm


def main(args):
    model_option = args.model_option
    dataset_name = args.dataset_name
    splits = args.splits
    sampling = args.sampling
    llm_option = args.llm_option

    result_path = f'../results/{model_option}_{dataset_name}_{splits}'
    if sampling:
        result_path += '_sampling'
    result_path += '.pk'

    output_path = f'../results/{llm_option.split("/")[-1]}_{model_option}_{dataset_name}'
    if sampling:
        output_path += '_sampling'
    output_path += '.pk'

    with open(result_path, 'rb') as f:
        cot_result_list = pickle.load(f)
    print(f'Loading cot results from {result_path}...')
    print(len(cot_result_list))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=args.api_key,
    )

    base_prompt = 'Select the most appropriate answer that can be concluded from the ' + \
                  'given rationale. You must choose only ONE answer. ' + \
                  'Directly output in the format of "Therefore, the answer is ...".'

    cot_result_list = cot_result_list[:2]
    raw_results = []
    for i in tqdm(range(len(cot_result_list))):
        o = cot_result_list[i]
        llm_outputs = []
        for j, cot_f in enumerate(o['cot_f']):
            prompt = o['source'] + '\nRationale: ' + cot_f + '\n\n' + base_prompt
            completion = client.chat.completions.create(
                model=llm_option,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                top_p=0.9,
                temperature=0.9
            )
            try:
                content = completion.choices[0].message.content
            except:
                content = 'ERROR.'

            content = content.split('\n\n')[0]
            llm_outputs.append(content)
        raw_results.append(llm_outputs)

    with open(output_path, 'wb') as f:
        pickle.dump(raw_results, f)


if __name__ == '__main__':
    '''
    Example usage:
        python run_llms.py \
            --model_option olmo-7B \
            --dataset_name commonsense_qa \
            --splits validation \
            --llm_option meta-llama/llama-3.3-70b-instruct \
            --api_key <YOUR_API_KEY> \
            --sampling
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_option', type=str, default='mistral-7B')
    parser.add_argument('--dataset_name', type=str, default='social_i_qa')
    parser.add_argument('--splits', type=str, default='validation')
    parser.add_argument('--sampling', action='store_true')
    parser.add_argument('--api_key', type=str, help='Your API key for openrouter.ai')
    parser.add_argument('--llm_option', type=str, default="meta-llama/llama-3.3-70b-instruct", choices=[
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini"
    ])
    args = parser.parse_args()

    main(args)