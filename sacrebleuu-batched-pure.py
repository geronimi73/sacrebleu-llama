# pure = use model.generate instead of transformers.pipeline

import transformers
import evaluate
import torch
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")

parser = argparse.ArgumentParser()

parser.add_argument("-m","--model_path", default="models/llama2-7b")
parser.add_argument("--lang", default="eng_Latn")
parser.add_argument("-4","--four_bit", action="store_true")
parser.add_argument("-8","--eight_bit", action="store_true")
parser.add_argument("--bs", type=int, default=12)

args=parser.parse_args()

model_path=args.model_path
lang=args.lang
bs=args.bs
load_4bit=args.four_bit
load_8bit=args.eight_bit

num_shots=5
data_set={
	"path": "wmt20_mlqe_task1",
	"name": "en-de",
	"split": "test",
}

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({"pad_token":"<pad>"})

nf4_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_use_double_quant=True,
	bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    quantization_config=nf4_config if load_4bit else None,
    load_in_8bit=load_8bit
    )
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

ds = load_dataset(**data_set)
prompt_template="English: {en}\nGerman: {de}"
if num_shots==0:
	prompt_examples=""
else:
	ds_examples=ds[0:num_shots]		# use first 5 to generate examples for 5-shot translation prompt
	prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples["translation"]])
ds_predict=ds[num_shots:]
prompts=[ (prompt_examples+"\n\n"+prompt_template).format(en=d["en"],de="")[:-1] for d in ds_predict["translation"] ] 

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
	"do_sample": True,
	"num_return_sequences": 1,
	"eos_token_id": tokenizer.eos_token_id,
	"max_new_tokens": 50,		
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

for start in tqdm(range(0,len(prompts),bs)):
	stop=min(start+bs,len(prompts)-1)
	prompts_batch=prompts[start:stop]

	encodings=tokenizer(prompts_batch, return_tensors="pt", padding='longest', truncation=False).to("cuda")
	with torch.no_grad():
		output_ids = model.generate(**encodings, **gen_config)
	outputs=tokenizer.batch_decode(output_ids, skip_special_tokens=True)

	for i,output in enumerate(outputs):
		sample_no=i+start

		prediction=output[len(prompts[sample_no]):]
		prediction=prediction.split("\n")[0].strip() if "\n" in prediction else prediction.strip()

		reference=ds_predict["translation"][sample_no]["de"]	
		original=ds_predict["translation"][sample_no]["en"]		

		results["translations"].append({"input": original, "reference": reference, "prediction": prediction})
		results["num_translations"]+=1

		sacrebleu_results=sacrebleu.compute(predictions=[t["prediction"] for t in results["translations"]], references=[t["reference"] for t in results["translations"]])
		results["sacrebleu_score"]=sacrebleu_results["score"]

		print(results["sacrebleu_score"])

		write_pretty_json("results_new/sacrebleu-" + model_path.split("/")[-1] + f"_{num_shots}-shot" + ".json",results)


