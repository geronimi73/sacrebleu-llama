import transformers
import evaluate
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import StoppingCriteria, StoppingCriteriaList
import argparse

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-1:])).item():
                return True
        return False

model_path="models/ALMA-13B"
data_set={
	"path": "wmt20_mlqe_task1",
	"name": "en-de",
	"split": "test",
}
num_shots=5

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

prompt_template="\nEnglish: {en}\nGerman: {de}"

if num_shots==0:
	prompt_examples=""
else:
	ds_examples=ds[0:num_shots]		# use first 5 to generate examples for 5-shot translation prompt
	prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples["translation"]])

ds_predict=ds[num_shots:]

prompts=[ (prompt_examples+"\n\n"+prompt_template).format(en=d["en"],de="")[:-1] for d in ds_predict["translation"] ] 
prompts_generator=(p for p in prompts)	# pipeline needs a generator, not a list


stop_words_ids = [pipeline.tokenizer("\n\n", return_tensors='pt',add_special_tokens=False)['input_ids'][0][-1]]			# tokenizer issue, llama tokenizer always prepends 29871 to \n, therefore generate \n\n and use last one
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
	"do_sample": True,
	"num_return_sequences": 1,
	"eos_token_id": pipeline.tokenizer.eos_token_id,
	"max_new_tokens": 500,		
    "stopping_criteria": stopping_criteria,		# stop on "\n"
}

results={
	"model": model_path,
	"dataset": data_set["path"] + "_" + data_set["name"],
	"gen_config": {k: gen_config[k] for k in set(list(gen_config.keys())) - set(["stopping_criteria"])},	# exclude stopping criteria, not json serializable
	"num_shots": num_shots,
	"num_translations": 0,
	"sacrebleu_score": None,
	"translations": [],
}

sacrebleu = evaluate.load("sacrebleu")

for i, out in enumerate(tqdm(pipeline(prompts_generator, batch_size=24, **gen_config),total=len(prompts))):
	prediction=out[0]["generated_text"][len(prompts[i]):].split("\n")[0].strip()

	reference=ds_predict["translation"][i]["de"]	
	original=ds_predict["translation"][i]["en"]		

	results["translations"].append({"input": original, "reference":reference, "prediction": prediction})
	results["num_translations"]+=1

	sacrebleu_results=sacrebleu.compute(predictions=[t["prediction"] for t in results["translations"]], references=[t["reference"] for t in results["translations"]])
	results["sacrebleu_score"]=sacrebleu_results["score"]

	print(results["sacrebleu_score"])
	write_pretty_json("results/sacrebleu-" + model_path.split("/")[-1] + f"_{num_shots}-shot" + ".json",results)


