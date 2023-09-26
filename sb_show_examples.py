import json
import random

def read_json(file_path):
    f = open(file_path)
    data = json.load(f)
    f.close()
    return data

transl_no=random.randrange(10,900)

for num_shots in range(11):
    input_fn=f"results/sacrebleu-llama2-7b_{num_shots}-shot.json"
    translations=read_json(input_fn)["translations"]

    print(str(num_shots) +"\t" + str(read_json(input_fn)["sacrebleu_score"]))
