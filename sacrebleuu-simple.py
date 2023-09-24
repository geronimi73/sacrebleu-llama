import transformers, evaluate, torch
from datasets import load_dataset
from tqdm import tqdm

pipeline  = transformers.pipeline("text-generation",
    model="models/llama2-7b",
    device_map="auto",
    torch_dtype=torch.bfloat16)

ds=load_dataset(path="wmt20_mlqe_task1", name="en-de",split="test")
ds=ds["translation"]

ds_examples=ds[0:5]
ds_predict=ds[5:]

prompt_template="English: {en}\nGerman: {de}"
prompt_examples = "\n\n".join([prompt_template.format(**row) for row in ds_examples])

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
for row in tqdm(ds_predict):
    prompt=prompt_examples + "\n\n" + prompt_template.format(en=row["en"], de="")[:-1]
    prediction=pipeline(prompt, **gen_config)[0]["generated_text"][len(prompt)+1:]

    if "\n" in prediction:
        prediction=prediction.split("\n")[0]
    predictions.append(prediction)

references=[row["de"] for row in ds_predict]

sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results=sacrebleu.compute(predictions=predictions, references=references)

print(sacrebleu_results["score"])
