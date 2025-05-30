## Dependency

Run the following command to install the required dependencies. Replace `mamba` with `conda` if mamba is not available.
```bash
mamba create -y -n biasedcot python=3.11
mamba activate biasedcot
bash build.sh
```

## Chain-of-thought Generation
To generate CoT, run the following command from the `./scripts` directory. This will generate zero-shot direction prediction for a given QA dataset, as well as the chain-of-thought reasoning (rationale and prediction) using `zeroshot_cot.txt` as prompt. Before running the script, make sure the huggingface permission for certain models are granted.

```bash
python cot.py \
  --model_option llama3-8B \
  --dataset_name commonsense_qa \
  --splits validation \
  --batch_size 8 \
  --n_output 1
```

If `--n_output` is set to 1, greedy decoding is used. Otherwise, sampling is used with `temperature=0.3` and `top_p=0.7` for llama3-8B, and `temperature=0.9` and `top_p=0.9` for mistral-7B and olmo2-7B.


## Extract \$A_{Inter}$ from CoT

To extract the intermediate answer from the generated CoT from the previous step, run the following command from the `./scripts` directory. 

```bash
python llms.py \
  --model_option mistral-7B \
  --dataset commonsense_qa \
  --splits validation \
  --llm_option openai/gpt-4o-mini \
  --api_key YOUR_OPENROUTER_API_KEY \
  --sampling
```

The script will search for the generated output files from the previous step based on `--model_option`, `--dataset`, `--splits`, and `--sampling`, and then extract the intermediate answer from the CoT using specified advanced LLMs. The script is run using [OpenRouter](https://openrouter.ai/) for LLMs' API access. The exact LLMs used in this work are listed below. Each LLM will predict answers based on the QA and the generated CoT from the previous step. Then, majority voting is used across multiple LLMs to approximate the true $A_{Inter}$. Hypothetically, one can specify any LLMs of interest.
- `meta-llama/llama-3.3-70b-instruct`
- `deepseek/deepseek-chat`
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4o-mini`


## Analysis of Confirmation Bias
Follow `./notebooks/analysis.ipynb` for the main analysis of confirmation bias. 