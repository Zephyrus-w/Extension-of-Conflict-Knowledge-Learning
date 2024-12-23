import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE=torch.device("cuda:0")

aspects = ['birth_date', 'birth_place', 'university', 'major', 'company', 'workplace']
#aspects = ['birth_date', 'birth_place', 'university', 'major', 'company', 'workplace', 'sat_score', 'paper_num']

QUESTION_LIST = {
    "birth_date": "When is {}'s birthday? {}.",
    "birth_place": "Where is {}'s birth place? {}.",
    "university": "Which university did {} attend? {}.",
    "major": "What is {}'s major? {}.",
    "company": "Which company did {} work in? {}.",
    "workplace": "Which city does {} work in? {}.",
    "sat_score": "What was {}'s score on the sat? {}.",
    "paper_num": "How many articles has {} posted? {}."

}
STATEMENT_LIST = {
    # "birth_date": "{} was born on {}.",
    "birth_date": "{}'s birthday is {}.",
    "birth_place": "{} was born at {}.",
    # "university": "The university that {} went to is {}.",
    "university": "{} received education at the {}.",
    "major": "{} focused on {} during her university study.",
    "company": "{} worked for {}.",
    "workplace": "The city {} worked in is {}.",
    "sat_score": "{} has a sat score of {}.",
    "paper_num": "{} has published {} papers."
}

def calculate_perplexity(sentence, model, tokenizer):
    model.eval()
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model(tensor_input.to(DEVICE), labels=tensor_input)[0]
    return torch.exp(loss.float()).item()

def answer_question(data_item, args, model, tokenizer):
    item_result = {
        "full_name": data_item["full_name"]
    }
    for aspect in aspects:
        prompt = STATEMENT_LIST[aspect]
        aspect_result = {}
        for source in args.source_list:
            statement = prompt.format(
                data_item["full_name"], 
                data_item[source][aspect])
            ppl = calculate_perplexity(statement, model, tokenizer)
            aspect_result[source] = {
                "statement": statement,
                "ppl": ppl
            }
        prefer_source = args.source_list[0] if aspect_result[args.source_list[0]]["ppl"] < aspect_result[args.source_list[1]]["ppl"] else args.source_list[1]
        aspect_result["prefer_source"] = prefer_source
        item_result[aspect] = aspect_result
    #item_result形如：
    # {
    #     "full_name":"xxx",
    #     "birth_date":{ 以下为aspect_result
    #         "New York Times":{
    #             "statement":"xxx",
    #             "ppl":xxx
    #         },
    #         "The Guardian":{
    #             "statement":"xxx",
    #             "ppl":xxx
    #         },
    #         "prefer_source":"xxx"
    #     },
    #     "..."
    # }
    return item_result

def calculate_type_acc(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    with open(args.test_info_truth, 'r', encoding='utf8') as f:
        truth_info = json.load(f)

    result_list = []
    if args.use_statement:
        prompt = STATEMENT_LIST[args.field]
    else:
        prompt = QUESTION_LIST[args.field]
    first_source_name = args.source_list[0]
    second_source_name = args.source_list[1]

    first_source_cnt = {
        "birth_date": 0,
        "birth_place": 0,
        "university": 0,
        "major": 0,
        "company": 0,
        "workplace": 0,
        # "sat_score": 0,
        # "paper_num": 0
    }
    second_source_cnt = {
        "birth_date": 0,
        "birth_place": 0,
        "university": 0,
        "major": 0,
        "company": 0,
        "workplace": 0,
        # "sat_score": 0,
        # "paper_num": 0
    }

    for data_item in tqdm(truth_info):
        result = answer_question(data_item, args, model, tokenizer)
        for aspect in aspects:
            if result[aspect]["prefer_source"] == first_source_name:
                first_source_cnt[aspect] += 1
            elif result[aspect]["prefer_source"] == second_source_name:
                second_source_cnt[aspect] += 1
            else:
                raise ValueError
        result_list.append(result)

    for aspect in aspects:
        print("choice {}: {}".format(first_source_name, first_source_cnt[aspect]))
        print("choice {}: {}".format(second_source_name, second_source_cnt[aspect]))
        print("the Pr(A,B) of {} is {}".format(aspect, first_source_cnt[aspect]/(first_source_cnt[aspect]+second_source_cnt[aspect])))
    
    #保存结果
    with open(args.result_path, 'w', encoding='utf8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    # parser.add_argument("--field")
    parser.add_argument("--use_statement", default = True, action="store_true")
    parser.add_argument("--test_info_truth",type = str, help = '测试数据真实值的路径')
    parser.add_argument("--source_list", nargs = '2', default = ['New York Times', 'The Guardian'], help = '传递文本特征列表')
    #python inference.py --source_list "New York Times" "The Guardian"
    parser.add_argument("--result_path", type = str, help = '结果保存路径')
    args = parser.parse_args()

    calculate_type_acc(args)
