
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 1286902
    num_examples: 4358
  - name: train
    num_bytes: 541646398
    num_examples: 1801350
  - name: validation
    num_bytes: 1147368
    num_examples: 3760
  download_size: 304529656
  dataset_size: 544080668
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
---

Output:
{
    "extracted_code": ""
}
