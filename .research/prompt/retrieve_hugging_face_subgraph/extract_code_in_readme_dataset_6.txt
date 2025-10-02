
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
- config_name: clean
  features:
  - name: text
    sequence: string
  splits:
  - name: train
    num_bytes: 830910180
    num_examples: 3805842
  - name: validation
    num_bytes: 1760396
    num_examples: 8139
  - name: test
    num_bytes: 1979602
    num_examples: 9421
  download_size: 311802693
  dataset_size: 834650178
- config_name: raw
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 519638274
    num_examples: 29567
  - name: validation
    num_bytes: 1103823
    num_examples: 60
  - name: test
    num_bytes: 1241003
    num_examples: 62
  download_size: 301810546
  dataset_size: 521983100
configs:
- config_name: clean
  data_files:
  - split: train
    path: clean/train-*
  - split: validation
    path: clean/validation-*
  - split: test
    path: clean/test-*
- config_name: raw
  data_files:
  - split: train
    path: raw/train-*
  - split: validation
    path: raw/validation-*
  - split: test
    path: raw/test-*
---

Output:
{
    "extracted_code": ""
}
