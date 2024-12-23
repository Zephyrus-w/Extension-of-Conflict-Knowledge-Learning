from src.data.sft_dataset import make_sft_data_module
from src.data.continue_pretrain import make_ct_data_module
def make_data_module(data_args,model_args,tokenizer,type):
    if type == "sft":
        return make_sft_data_module(data_args,model_args,tokenizer)
    elif type =="ct":
        return make_ct_data_module(data_args,model_args,tokenizer)