pip install transformers transformers_stream_generator tiktoken transformer_lens einops jaxtyping colorama scikit-learn

import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float, Int
from colorama import Fore

torch.set_grad_enabled(False)

"""### Load harmful/harmless prompts datasets, and chat template
Please note you may need to change the `CHAT_TEMPLATE` for whichever model you're using
"""

def get_jailbroken_suffix_instructions():
    dataset = load_dataset("Oluwadara/JBB-Behaviors-harmful-nanogcg", split="train")
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    return train, test

def get_refusal_instructions():
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    return train, test

refusal_instructions_train, refusal_instructions_test = get_refusal_instructions()
jailbroken_instructions_train, jailbroken_instructions_test = get_jailbroken_suffix_instructions()

CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>{{ system_prompt }}<</SYS>>{{ user_message }} [/INST]" 
    ) 

MODEL_PATH =  "meta-llama/Llama-2-7b-chat-hf"  

model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    dtype=torch.bfloat16,   
)


model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token = model.tokenizer.eos_token

"""#### Load model utility stuff"""

def tokenize_instructions_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    prompts = [CHAT_TEMPLATE.format(user=instruction) for instruction in instructions]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

tokenize_instructions_fn = functools.partial(tokenize_instructions_chat, tokenizer=model.tokenizer)
def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:
    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens
    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    generations = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations

try:
    del refusal_logits
except Exception:
    pass
try:
    del jailbroken_logits
except Exception:
    pass
gc.collect(); torch.cuda.empty_cache()

"""## Finding potential "refusal directions" (batched)"""

harmful = {}
harmless = {}

N_INST_TRAIN = 4

refusal_instructions_train, refusal_instructions_test = get_refusal_instructions()
jailbroken_instructions_train, jailbroken_instructions_test = get_jailbroken_suffix_instructions()

refusal_inst_train = [x['goal'] for x in refusal_instructions_train]
refusal_inst_test = [x['goal'] for x in refusal_instructions_test]

jailbroken_inst_train = [x['message']+ " " + x['result'] for x in jailbroken_instructions_train]
jailbroken_inst_test = [x['message']+ " " + x['result'] for x in jailbroken_instructions_test]

toks = tokenize_instructions_fn(instructions=refusal_inst_train[:N_INST_TRAIN]+jailbroken_inst_train[:N_INST_TRAIN])
refusal_toks,jailbroken_toks = toks.split(N_INST_TRAIN)

batch_size = 2

for i in tqdm(range(0, N_INST_TRAIN // batch_size + (N_INST_TRAIN % batch_size > 0))):
    id = i*batch_size
    e = min(N_INST_TRAIN,id+batch_size)

    # run the models on harmful and harmless prompts, cache their activations separately.
    refusal_logits, refusal_cache = model.run_with_cache(refusal_toks[id:e], names_filter=lambda hook_name: 'resid' in hook_name, device='cpu', reset_hooks_end=True)
    jailbroken_logits, jailbroken_cache = model.run_with_cache(jailbroken_toks[id:e], names_filter=lambda hook_name: 'resid' in hook_name, device='cpu', reset_hooks_end=True)

    for key in refusal_cache:
        if key not in refusal:
            refusal[key] = [refusal_cache[key]]
            jailbroken[key] = [jailbroken_cache[key]]
        else:
            refusal[key].append(refusal_cache[key])
            jailbroken[key].append(jailbroken_cache[key])

    # force Python & PyTorch to clear GPU and CPU RAM where possible
    del refusal_logits, refusal_logits, jailbroken_cache, jailbroken_cache
    gc.collect()
    torch.cuda.empty_cache()

refusal = {k:torch.cat(v) for k,v in refusal.items()}
jailbroken = {k:torch.cat(v) for k,v in jailbroken.items()}

"""### Compute activations into refusal directions"""

# compute difference of means between harmful and harmless activations at intermediate layers

def get_act_idx(cache_dict, act_name, layer):
    key = (act_name, layer,)
    return cache_dict[utils.get_act_name(*key)]

activation_layers = ['resid_pre', 'resid_mid', 'resid_post']

activation_refusals = {k:[] for k in activation_layers}

for layer_num in range(1,model.cfg.n_layers):
    pos = -1

    for layer in activation_layers:
        refusal_mean_act = get_act_idx(refusal, layer, layer_num)[:, pos, :].mean(dim=0)
        jailbroken_mean_act = get_act_idx(jailbroken, layer, layer_num)[:, pos, :].mean(dim=0)

        refusal_dir = jailbroken_mean_act - refusal_mean_act
        refusal_dir = refusal_dir / refusal_dir.norm()
        activation_refusals[layer].append(refusal_dir)

# save to file so you don't have to re-build later
torch.save(activation_refusals, 'refusal_dirs.pth')
refusal_dirs = activation_refusals


activation_refusals = torch.load('refusal_dirs.pth')
refusal_dirs = activation_refusals

"""## Ablate "refusal direction" via inference-time intervention

Given a "refusal direction" $\widehat{r} \in \mathbb{R}^{d_{\text{model}}}$ with unit norm, we can ablate this direction from the model's activations $a_{l}$:
$${a}_{l}' \leftarrow a_l - (a_l \cdot \widehat{r}) \widehat{r}$$

By performing this ablation on all intermediate activations, we enforce that the model can never express this direction (or "feature").

### "Score" layer activation diffs
This is a place with room for improvement in methodology. For now, I'm just doing a rudimentary sort based on difference distance average to find dirs with the most "change"
"""

# Get all calculated potential refusal dirs, sort them in Descending order (reverse) based on their mean()
activation_layers = ['resid_pre', 'resid_mid', 'resid_post'] # you can use a subset of these if you don't think certain activations are promising

activation_layers = ['resid_pre'] # this is usually good enough, though if you've got the compute to spare...
activation_scored = sorted([activation_refusals[layer][l-1] for l in range(1,model.cfg.n_layers) for layer in activation_layers], key = lambda x: abs(x.mean()), reverse=True)

"""#### Model ablation testing/brute-forcing the best refusal dir

##### Inference-time intervention hook:
"""

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

"""##### Testing baseline, can skip if you don't care (it will basically just be refusals with a regular model :P)"""

N_INST_TEST = 4
baseline_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=[])
for gen in baseline_generations:
    print(gen)

"""##### Evaluating layers defined earlier (needs human evaluation to determine best layer for refusal inhibition)"""

if "N_INST_TEST" not in locals() or not N_INST_TEST:
    N_INST_TEST = 4 # you may want to evaluate more at the cost of additional compute time. by default, batches are size of 4, so I'd recommend making it a multiple of 4.
EVAL_N = 20 # Evaluate how many of the top N potential dirs
evals = []
for refusal_dir in tqdm(activation_scored[:EVAL_N]):
    intervention_layers = list(range(model.cfg.n_layers)) # all layers

    hook_fn = functools.partial(direction_ablation_hook,direction=refusal_dir)
    fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

    intervention_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=fwd_hooks)
    evals.append(intervention_generations)

    #print(intervention_generations) # if you want to watch it as it goes

"""#### Present evals to clever pre-trained non-refusing human"""

for instruction in range(N_INST_TEST):
    if 'baseline_generations' in locals() and baseline_generations and len(baseline_generations) > instruction:
        print(f"INSTRUCTION {instruction}: {repr(harmful_inst_test[instruction])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_generations[instruction]), width=100, initial_indent='\t', subsequent_indent='\t'))
    for layer_candidate in range(EVAL_N):
        if len(evals) > layer_candidate and len(evals[layer_candidate]) > instruction:
            print(Fore.RED + f"LAYER CANDIDATE #{layer_candidate} INTERVENTION COMPLETION:")
            print(textwrap.fill(repr(evals[layer_candidate][instruction]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)

"""## Orthogonalize weights w.r.t. "refusal direction"

We can implement the intervention equivalently by directly orthogonalizing the weight matrices that write to the residual stream with respect to the refusal direction $\widehat{r}$:
$$W_{\text{out}}' \leftarrow W_{\text{out}} - \widehat{r}\widehat{r}^{\mathsf{T}} W_{\text{out}}$$

By orthogonalizing these weight matrices, we enforce that the model is unable to write direction $r$ to the residual stream at all!

This is basically how you finalize your layers' weights to represent your orthogonalization for a saved model

### Choose your fighter (favorite, ideally non-refusing layer)
"""

layer_candidate = 8 # e.g. you should choose based on the layer you think aligns to the behavior you like
refusal_dir = activation_scored[layer_candidate]

"""### Write ortho'd weights into model"""

def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    proj = einops.einsum(matrix, vec.view(-1, 1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj

if refusal_dir.device != model.W_E.device:
    refusal_dir = refusal_dir.to(model.W_E.device)
model.W_E.data = get_orthogonalized_matrix(model.W_E, refusal_dir)

for block in tqdm(model.blocks):
    if refusal_dir.device != block.attn.W_O.device:
        refusal_dir = refusal_dir.to(block.attn.W_O.device)
    block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O, refusal_dir)
    block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out, refusal_dir)

# save your refusal_dir of choice separately to a file
torch.save(refusal_dir,"ablation.pth")

"""### Verify model weights are adjusted to match ablation (skippable)"""

orthogonalized_generations = get_generations(model, harmful_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=[])

for i in range(N_INST_TEST):
    if 'baseline_generations' in locals() and baseline_generations and len(baseline_generations) > i:
        print(f"INSTRUCTION {i}: {repr(harmful_inst_test[i])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RED + f"INTERVENTION COMPLETION:")
    print(textwrap.fill(repr(evals[layer_candidate][i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.MAGENTA + f"ORTHOGONALIZED COMPLETION:")
    print(textwrap.fill(repr(orthogonalized_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)

"""## Save your unruly model
This is where you'll need to consult with the original structure of the model you're generating. Below is converting Phi-3 and Llama-3 examples, but you'll need to do differently for others. Or if you just want a "no thinking" save, you can use the pytorch save below. Be aware that the structure output by that is not directly convertable however.

### Simple PyTorch save! (easiest, but least portable)
"""

torch.save(model, "pytorch_model.bin") # can name it whatever you want, and then reload it

"""### Converting models back to HF safetensors (harder)
Do note that we only adjust a couple layers in get_orthogonalized_matrix(), so only need to copy 1 + (2*n_layers) over to the original trained model, not the whole lot.

You can look to TransformerLens's source code `loading_from_pretrained.py` to see how the layers get converted in. e.g. https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py#L1746-L1833 is `convert_llama_weights`, so you can just reverse the steps for the layers that you alter

References to convert functions per model:
https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py#L1475-L1504
"""

# this is probably useful for any conversion
cfg = model.cfg

state_dict = model.state_dict()

hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16) # load the original model as a regular unhooked Transformer -- don't need to load it into GPU as it's just for saving
lm_model = hf_model.model

"""#### Llama-3 conversion"""

lm_model.embed_tokens.weight = torch.nn.Parameter(state_dict["embed.W_E"].cpu())

for l in range(cfg.n_layers):
    lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(einops.rearrange(state_dict[f"blocks.{l}.attn.W_O"], "n h m->m (n h)", n=cfg.n_heads).contiguous())
    lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"],0,1).contiguous())

"""#### Phi-3 conversion"""

lm_model.embed_tokens.weight = state_dict["embed.W_E"]

for l in range(cfg.n_layers):

    W_O = einops.rearrange(
        state_dict[f"blocks.{l}.attn.W_O"], "n_head d_head d_model -> d_model (n_head d_head)", n_head=cfg.n_heads
    )
    lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(W_O.contiguous())

    lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"].cpu(), 0, 1).contiguous())

"""#### Save converted model"""

hf_model.save_pretrained("path/to/my/based-model")