
> #### `install.sh` executes the following commands automatically! Please try it first.

## Manual Installation Instructions

Install RL-X
```bash
conda create -n trirl python=3.11.4
conda activate trirl
git clone https://github.com/nico-bohlinger/RL-X.git && cd RL-X && git checkout 14379ff # clone a specific commit to ensure reproducibility
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install -U "jax[cuda12]"
```

Install this package 
```bash
git clone https://github.com/anishhdiwan/trust-region-irl
cd trust-region-irl
pip install -e .
```

Install LocoMujoco
```bash
cd ..
git clone https://github.com/robfiras/loco-mujoco.git && cd loco-mujoco && git checkout 131c1e7 # clone a specific commit to ensure reproducibility
pip install -e . 
```

#### Dataset

We host the dataset on **HuggingFace [https://huggingface.co/datasets/anishdiwan/trirl_dataset](https://huggingface.co/datasets/anishdiwan/trirl_dataset)**.
```bash
# Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/datasets/anishdiwan/trirl_dataset
```