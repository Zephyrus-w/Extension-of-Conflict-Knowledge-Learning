# how to alter the files into another server?

1. ~/data/concat_data_source_names.py

    line 72~82: alter the raw material file path. In ~/data/raw_data_item

    line 109, 111, 203, 206 (and 208): alter the output file path. In ~/data/output_info or ~/data/output_text

    maybe: line 20: newspaper list


2. ~/inference/inference.py

    maybe: line 12, 23: STATEMENT_LIST, QUESTION_LIST

    maybe: line 9: aspects


3. ~/inference/inference.bash
    

use LLaMA-2-7b-hf as model.

if model downloaded from meta, then use transformers to convert weights to hf version.

#!/bin/bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /home/bingxing2/home/scx7002/model/Meta-Llama-2-7b-raw \
    --model_size 7B \
    --output_dir /home/bingxing2/home/scx7002/model/Meta-Llama-2-7B-hf