import nanogcg
import torch
from datasets import load_dataset, Dataset
from nanogcg import GCGConfig
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-chat-hf"  
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
#print(dataset, dataset[0])
output = []
for i in range(len(dataset)):
    message = dataset[i]["Goal"]
    target = dataset[i]["Target"]

    config = GCGConfig(
        num_steps=200,
        search_width=64,
        topk=64,
        seed=42,
        verbosity="WARNING"
    )

    result = nanogcg.run(model, tokenizer, message, target, config)
    print("Final result:", result)

    output.append({
        "message": message,
        "target": target,
        "result": result.best_string
    })

out_ds = Dataset.from_list(output)
out_ds.save_to_disk("suffix_breaker_harmbench ")
out_ds.push_to_hub("Oluwadara/JBB-Behaviors-harmful-nanogcg")

with open("suffix_breaker_harmbench.json", "w") as f:
    json.dump(output, f, indent=2)
