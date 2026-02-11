# Source code for paper "Zeroth-Order Fine-Tuning of LLMs in Random Subspaces"

This is the implementation for the paper [Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](http://arxiv.org/abs/2410.08989). 

In this paper, we propose the random Subspace Zeroth-order (SubZero) optimization to address the challenges posed by LLMs’ high dimensionality. We introduce a low-rank perturbation tailored for LLMs that significantly reduces memory consumption while improving training performance. Additionally, we have successfully applied SubZero to four popular fine-tuning schemes for LLMs, including full parameter tuning, LoRA, prefix tuning, and prompt tuning. This demonstrates SubZero's compatibility and versatility across different tuning approaches. 

Furthermore, we prove that our gradient estimation closely approximates the backpropagation gradient, exhibits lower variance than traditional ZO methods, and ensures convergence when combined with SGD. Experimental results show that SubZero enhances fine-tuning performance and achieves faster convergence compared to standard ZO approaches like [MeZO](https://github.com/princeton-nlp/MeZO) across various language modeling tasks.


<p>
  <img src="./figure/subzero.png?raw=true" alt="Fig" width="100%"/>
  <em>
    Visualization of cosine similarity, relative variance, training loss and GPU memory cost on OPT-1.3B under the prompt tuning scheme. SubZero demonstrates reduced angle error and variance in gradient estimation, while also accelerating convergence with minimal additional memory overhead.
  </em>
</p>

## Getting start
- We use python 3.10 and torch 2.1.0, transformers 4.28.1, and cuda 11.8.0.
- pip install -r requirements.txt

## Usage

Use `run.py` for all functions (zero-shot/ICL/fine-tuning/MeZO/SubZero):
```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments. We introduce some of the most important ones below. 
* `--num_train`: Number of training examples. For ICL, this is the number of demonstrations.
* `--num_dev`: Number of validation examples.
* `--num_test`: Number of testing examples.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot/ICL), `regular` (fine-tuning), `zo_sgd` (MeZO), `zo_adamu` (AdaMU), `subzero_sgd` (SubZero), or `subzero_adamu` (SubZO-AdaMU).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
* `--zo_eps`: ZO hyperparameter epsilon
* `--gauss_rank`, `--update_interval`: SubZero subspace rank and refresh interval (used by `subzero_sgd` and `subzero_adamu`).
* `--zo_adamu_*`: AdaMU schedule/hyperparameters (`zo_adamu_T1/T2/T3`, `zo_adamu_alpha_target`, `zo_adamu_beta1_target`, `zo_adamu_beta2_target`, `zo_adamu_sigma`).
* `--prefix_tuning`: use prefix-tuning. 
* `--lora`: use LoRA.
* `--prompt_tuning`: use prompt-tuning.

## Reproducing Results

We provide an example of the OPT-1.3b model performing prompt tuning on the SST-2 dataset.

### MeZO-SGD
`CUDA_VISIBLE_DEVICES=0 python run.py --task_name=SST2 --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-SST2-prompt-mezo --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1    --eval_steps=1000     --max_steps=20000  --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=zo_sgd    --train_set_seed=0 --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --prompt_tuning --num_virtual_tokens=10 --prompt_init_by_real_tokens --learning_rate=1e-3 --zo_eps=1e-2     --weight_decay=0`

`CUDA_VISIBLE_DEVICES=0 python run.py --task_name=SST2 --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-SST2-prompt-adamu --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1    --eval_steps=1000     --max_steps=20000  --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=zo_adamu    --train_set_seed=0 --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --prompt_tuning --num_virtual_tokens=10 --prompt_init_by_real_tokens --learning_rate=1e-3 --zo_eps=1e-2     --weight_decay=0`

`CUDA_VISIBLE_DEVICES=0 python run.py --task_name=SST2 --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-SST2-prompt-muon --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1    --eval_steps=1000     --max_steps=20000  --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=zo_muon    --train_set_seed=0 --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --prompt_tuning --num_virtual_tokens=10 --prompt_init_by_real_tokens --learning_rate=1e-3 --zo_eps=1e-2     --weight_decay=0`

### SubZero-SGD
`CUDA_VISIBLE_DEVICES=0 python run.py --task_name=SST2 --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-SST2-prompt-subzero --num_train_epochs=5    --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=subzero_sgd --train_set_seed=0     --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --prompt_tuning --num_virtual_tokens=10 --prompt_init_by_real_tokens  --learning_rate=1e-3     --zo_eps=1e-2 --weight_decay=0 --gauss_rank=24 --update_interval=1000`

dolly
 `python run.py   --task_name=DOLLY   --model_name=facebook/opt-1.3b   --output_dir=result/opt1.3b-DOLLY-prompt-subzero   --max_steps=20000   --per_device_train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-4   --zo_eps=3e-3   --weight_decay=0   --trainer=subzero_sgd   --perturbation_mode=two_side   --lr_scheduler_type=constant   --logging_steps=10   --evaluation_strategy=steps   --eval_steps=500   --save_strategy=steps   --save_steps=500   --save_total_limit=1   --load_best_model_at_end   --num_train=2000   --num_dev=200   --num_eval=200   --train_set_seed=0   --prompt_tuning   --num_virtual_tokens=10   --prompt_init_by_real_tokens   --max_new_tokens=128   --num_beams=1 --dolly_data_path /root/autodl-tmp/FederatedScope-1/data/databricks-dolly-15k.jsonl --max_length 512`

### FO-SGD
`CUDA_VISIBLE_DEVICES=0 python run.py --task_name=SST2 --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-SST2-prompt-sgd --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=sgd --optimizer=sgd --train_set_seed=0 --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --prompt_tuning --num_virtual_tokens=10 --prompt_init_by_real_tokens --learning_rate=1e-3 --zo_eps=1e-2 --weight_decay=0`

## read tfevent

`python -u read_tfevents.py "large_models/result/SST2/opt-1.3b/prompt/subzo_adamu/subzo_adamu-SST2-0-opt-1.3b-OPTIM_prompt-STEP20000-adamw-momen0.0-LR0.001-constant-ZOEPS0.01-T2000-gauss_rank8-Q1-bs16-gradAccumulation1/2026-02-07_22-23-28/events.out.tfevents.1770474209.autodl-container-e999448ba2-4300a65f.3553.0" --scalars accuracy/val accuracy/test train_loss --summary max`

## Acknowledgment

This project is built upon the foundation laid by [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://github.com/princeton-nlp/MeZO) and [Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark](https://github.com/ZO-Bench/ZO-LLM/tree/main). The original code from their project is licensed under the [MIT License](https://github.com/princeton-nlp/MeZO/blob/main/LICENSE) and [License](https://github.com/ZO-Bench/ZO-LLM/blob/main/LICENSE) respectively. We would like to thank the authors for their great work and contributions.
