from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import argparse

def calculate_perplexity(model, tokenizer,sents):
    model.eval()  # make sure the model is in evaluation mode
    total_loss = 0
    total_examples = 0

    with torch.no_grad():  # we don't need to calculate gradients
        for sent in sents:
            encoding = tokenizer(sent,return_tensors="pt").to("cuda")
            inputs, labels = encoding["input_ids"], encoding["input_ids"].clone()
            outputs = model(input_ids=inputs, labels=labels)
            total_loss += outputs.loss.item() * inputs.size(0)
            total_examples += inputs.size(0)

    average_loss = total_loss / total_examples
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True,use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,low_cpu_mem_usage=True).cuda().half()
    model.eval()

    with torch.no_grad():
        encoding = tokenizer(args.input,return_tensors="pt").to("cuda")

        input_ids = encoding["input_ids"]
        out_ids = model.generate(
            **{"input_ids":input_ids},
            max_new_tokens=256,
            num_beams=1,
            do_sample=False

        )

        completion = tokenizer.batch_decode(out_ids,skip_special_tokens=True)[0]
        print(completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--input")

    args = parser.parse_args()
    main(args)