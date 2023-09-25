from transformers import AutoModelForCausalLM

import transformers, evaluate, torch
from datasets import load_dataset
from tqdm import tqdm

model_path="models/llama2-70b"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

ds=load_dataset(path="wmt20_mlqe_task1", name="en-de",split="test")
ds=ds["translation"]

ds_examples=ds[0:5]
ds_predict=ds[5:]

prompt_template="English: {en}\nGerman: {de}"
prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples])
prompts=[ (prompt_examples + "\n\n" + prompt_template).format(en=d["en"],de="")[:-1] for d in ds_predict["translation"] ] 

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
    "do_sample": True,
    "max_new_tokens": 100,  
    "pad_token_id": pipeline.tokenizer.eos_token_id,
}

predictions=[]
for i, prompt in enumerate(tqdm(ds_predict)):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output_ids = model.generate(inputs.input_ids, **gen_config)
    prediction=tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0][len(prompt)+1:]
    prediction=prediction.split("\n")[0] if "\n" in prediction else prediction
        
    predictions.append(prediction)

references=[row["de"] for row in ds_predict]

sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results=sacrebleu.compute(predictions=predictions, references=references)

print(sacrebleu_results["score"])
