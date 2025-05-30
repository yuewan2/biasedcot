mamba install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers datasets huggingface_hub accelerate
pip install nltk numpy==1.26.4 scipy scikit-learn
pip install openai
pip install -e .