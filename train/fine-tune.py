import json
import torch
from torch.utils.data import Dataset, DataLoader
import deepspeed
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from typing import List
import argparse
import pynvml

def print_gpu_info():
    """打印每个GPU的内存使用情况"""
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    print(f"\n当前系统中有 {deviceCount} 块GPU:")
    
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i}:")
        print(f"总内存: {info.total / 1024**2:.2f} MB")
        print(f"已用内存: {info.used / 1024**2:.2f} MB")
        print(f"剩余内存: {info.free / 1024**2:.2f} MB\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="替换成实际的 llama-2-7b 路径或huggingface的repo名称")
    parser.add_argument("--train_file_path", type=str, help="训练数据的路径")
    parser.add_argument("--output_dir", type=str, default="./output_llama_finetuned")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--deepspeed_config", type=str, default="conf.json")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    return parser.parse_args()

class MyTextDataset(Dataset):
    """
    一个简单的 Dataset 示例，读取形如 [{"text": "..."}] 的 JSON 数据文件。
    """
    def __init__(self, file_path: str, tokenizer: LlamaTokenizer, max_length: int = 512):
        super().__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取数据文件
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)  # 形如: [{"text": "..."}]
        
        for item in json_data:
            self.data.append(item["text"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        return text


def collate_fn(examples: List[str], tokenizer: LlamaTokenizer, max_length: int = 512):
    """
    collate 函数，将文本列表转换为模型可直接使用的 input_ids 和 attention_mask。
    """
    tokenized = tokenizer(
        examples,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return tokenized


def main(args):
    # -----------------------------
    # 1. 配置部分（可根据需要自行修改）
    # -----------------------------

    # 是否使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 2. 准备分词器与数据集
    # -----------------------------
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    # Llama 分词器没有 pad_token，需要手动指定特殊 token
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = MyTextDataset(
        file_path=args.train_file_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    # DataCollator 用于对输入进行 masked language modeling（如果是纯 causal LM，可不需要 MLM）
    # 这里为了演示，直接使用 DataCollatorForLanguageModeling，无监督任务通常是 causal LM；如有需要可自行修改。
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM 设置为 False
    )

    # 注意：如果使用 DeepSpeed 的 ZeRO 并行，需要保证 drop_last=False 使所有 rank 收到一样大小的 batch
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length),
        drop_last=False
    )

    print_gpu_info()

    # -----------------------------
    # 3. 加载模型
    # -----------------------------
    print("\n开始从HF格式的文件夹加载初始模型到CPU...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        # device_map='cpu',          # 显式指定先加载到 CPU
        torch_dtype=torch.float16, # 如果你的 GPU 能支持半精度可用，否则换成 float32
        low_cpu_mem_usage=True     # 进一步优化 CPU 占用，可选
    )

    # 这里假设你已经加载/初始化好 tokenizer
    model.resize_token_embeddings(len(tokenizer))
    '''
    print("\n将要model.to(device)")
    print_gpu_info()  # 你代码里已有的函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n将模型从CPU移动到GPU...")
    model = model.to(device)
    print("\n模型已加载到 GPU 上。")
    '''
    # -----------------------------
    # 4. 构建优化器与调度器
    # -----------------------------
    # 这里示例使用 AdamW，可根据需要自行调参
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": args.weight_decay
        }
    ]
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=args.learning_rate)

    # 学习率调度器
    total_steps = len(train_dataloader) * args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    print("\n优化器与调度器已构建。")
    # -----------------------------
    # 5. 初始化 DeepSpeed 引擎
    # -----------------------------
    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        training_data=train_dataset,  # 如果使用 dataset 方式，也可以不显式传 train_dataloader
        config=args.deepspeed_config,
        lr_scheduler=lr_scheduler
    )
    if model_engine.local_rank == 0:
        print("\nDeepSpeed初始化后的GPU使用情况：")
        print_gpu_info()
    print("\nDeepSpeed 引擎已初始化。")
    print(f"模型加载在以下设备上: {model_engine.device}")

    # -----------------------------
    # 6. 开始训练
    # -----------------------------
    print("\n开始训练...")
    global_step = 0
    for epoch in range(args.num_train_epochs):
        # 每个epoch开始前打印GPU情况
        if model_engine.local_rank == 0:
            print(f"\nEpoch {epoch} 开始时的GPU使用情况：")
            print_gpu_info()
        model_engine.train()
        for batch in train_dataloader:
            # 将输入移动到 GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 计算loss
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss

            # 反向传播与优化
            model_engine.backward(loss)
            model_engine.step()

            # 更新 global_step（仅用于打印或日志）
            global_step += 1
            if global_step % 100 == 0 and model_engine.local_rank == 0:
                print(f"Epoch {epoch}, step {global_step}, loss: {loss.item()}")

    # -----------------------------
    # 7. 保存模型
    # -----------------------------
    # 仅在 rank 0 的进程上进行保存
    if model_engine.local_rank == 0:
        # DeepSpeed Engine 可以保存最新状态
        model_engine.save_checkpoint(args.output_dir)

        # 如果只想保存 transformer 权重并与HuggingFace兼容，可以：
        # 注意：当使用 ZeRO-2 / ZeRO-3 时，需要调用 
        #      model_engine.module 去获取原始模型
        model_engine.module.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)