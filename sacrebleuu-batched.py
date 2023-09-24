import transformers
import evaluate
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
import argparse

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('num_shots', type=int)
parser.add_argument('model', type=str)

args = parser.parse_args()

num_shots=args.num_shots
model_path=args.model

# model_path="models/llama2-7b"
data_set={
	"path": "wmt20_mlqe_task1",
	"name": "en-de",
	"split": "test",
}
# num_shots=0

pipeline = transformers.pipeline(
		"text-generation",
		model=model_path,
		torch_dtype=torch.bfloat16,
		device_map="auto",
	)

# needed for batching, from "tips" at https://huggingface.co/docs/transformers/model_doc/llama2
pipeline.tokenizer.add_special_tokens({"pad_token":"<pad>"})
pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))
pipeline.model.config.pad_token_id = pipeline.tokenizer.pad_token_id

ds = load_dataset(**data_set)

prompt_template="English: {en}\nGerman: {de}"

if num_shots==0:
	prompt_examples=""
else:
	ds_examples=ds[0:num_shots]		# use first 5 to generate examples for 5-shot translation prompt
	prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples["translation"]])

ds_predict=ds[num_shots:]

prompts=[ (prompt_examples+"\n\n"+prompt_template).format(en=d["en"],de="")[:-1] for d in ds_predict["translation"] ] 
prompts_generator=(p for p in prompts)	# pipeline needs a generator, not a list

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
	"do_sample": True,
	"num_return_sequences": 1,
	"eos_token_id": pipeline.tokenizer.eos_token_id,
	"max_new_tokens": 100,		# sentences are short, this should be more than enough
}

results={
	"model": model_path,
	"dataset": data_set["path"] + "_" + data_set["name"],
	"gen_config": gen_config,
	"num_shots": num_shots,
	"num_translations": 0,
	"sacrebleu_score": None,
	"translations": [],
}

sacrebleu = evaluate.load("sacrebleu")

for i, out in enumerate(tqdm(pipeline(prompts_generator, batch_size=56, **gen_config),total=len(prompts))):
	prediction=out[0]["generated_text"][len(prompts[i])+1:].split("\n")[0]

	# ! change for new language
	reference=ds_predict["translation"][i]["de"]	
	original=ds_predict["translation"][i]["en"]		

	results["translations"].append({"input": original, "reference":reference, "prediction": prediction})
	results["num_translations"]+=1

	sacrebleu_results=sacrebleu.compute(predictions=[t["prediction"] for t in results["translations"]], references=[t["reference"] for t in results["translations"]])
	results["sacrebleu_score"]=sacrebleu_results["score"]

	print(results["sacrebleu_score"])
	write_pretty_json("sacrebleu-" + model_path.split("/")[-1] + f"_{num_shots}-shot" + ".json",results)


