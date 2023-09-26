from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

import transformers, evaluate, torch
from datasets import load_dataset
from tqdm import tqdm
import json

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")

model_path="models/llama2-70b"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

data_set={
    "path": "wmt20_mlqe_task1",
    "name": "en-de",
    "split": "test",
}

ds=load_dataset(**data_set)
ds=ds["translation"]

ds_examples=ds[0:5]
ds_predict=ds[5:]

prompt_template="English: {en}\nGerman: {de}"
prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples])
prompts=[(prompt_examples + "\n\n" + prompt_template).format(en=d["en"],de="")[:-1] for d in ds_predict] 

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 100,  
    "pad_token_id": tokenizer.eos_token_id,
}

results={
    "model": model_path,
    "dataset": data_set["path"] + "_" + data_set["name"],
    "gen_config": gen_config,
    "num_shots": 5,
    "num_translations": 0,
    "sacrebleu_score": None,
    "translations": [],
}


predictions=[]
references=[]
sacrebleu = evaluate.load("sacrebleu")

for i, prompt in enumerate(tqdm(prompts)):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output_ids = model.generate(inputs.input_ids, **gen_config)
    prediction=tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0][len(prompt)+1:]
    prediction=prediction.split("\n")[0] if "\n" in prediction else prediction
        
    predictions.append(prediction)
    references.append(ds_predict[i]["de"])

    sacrebleu_results=sacrebleu.compute(predictions=predictions, references=references)

    results["translations"].append({"input": ds_predict[i]["en"], "reference": ds_predict[i]["de"], "prediction": prediction})
    results["num_translations"]+=1
    results["sacrebleu_score"]=sacrebleu_results["score"]

    print(sacrebleu_results["score"])
    write_pretty_json("sacrebleu-" + model_path.split("/")[-1] + f"_{5}-shot" + ".json",results)

