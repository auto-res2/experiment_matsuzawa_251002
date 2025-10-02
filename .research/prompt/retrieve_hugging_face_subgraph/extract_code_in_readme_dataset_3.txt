
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': airplane
          '1': automobile
          '2': bird
          '3': cat
          '4': deer
          '5': dog
          '6': frog
          '7': horse
          '8': ship
          '9': truck
  splits:
  - name: train
    num_bytes: 113648310.0
    num_examples: 50000
  - name: test
    num_bytes: 22731580.0
    num_examples: 10000
  download_size: 143650937
  dataset_size: 136379890.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

Output:
{
    "extracted_code": ""
}
