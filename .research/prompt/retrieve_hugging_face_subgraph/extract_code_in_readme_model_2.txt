
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README

---
tags:
- image-classification
- ecology
- fungi
- FGVC
library_name: DanishFungi
license: cc-by-nc-4.0
---
# Model card for BVRA/resnet18.in1k_ft_df20_299

## Model Details
- **Model Type:** Danish Fungi Classification 
- **Model Stats:**
  - Params (M): 12.0M
  - Image size: 299 x 299
- **Papers:**
  - **Original:** Deep Residual Learning for Image Recognition --> https://arxiv.org/pdf/1512.03385
  - **Train Dataset:** DF20 --> https://github.com/BohemianVRA/DanishFungiDataset/

## Model Usage
### Image Embeddings
```python
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from urllib.request import urlopen
model = timm.create_model("hf-hub:BVRA/resnet18.in1k_ft_df20_299", pretrained=True)
model = model.eval()
train_transforms = T.Compose([T.Resize((299, 299)), 
                              T.ToTensor(), 
                              T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
img = Image.open(PATH_TO_YOUR_IMAGE)
output = model(train_transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor
# output is a (1, num_features) shaped tensor
```

## Citation 
```bibtex
@InProceedings{Picek_2022_WACV,
    author    = {Picek, Lukas and Sulc, Milan and Matas, Jiri and Jeppesen, Thomas S. and Heilmann-Clausen, Jacob and L{e}ss{\o}e, Thomas and Fr{\o}slev, Tobias},
    title     = {Danish Fungi 2020 - Not Just Another Image Recognition Dataset},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {1525-1535}
}

@article{picek2022automatic,
  title={Automatic Fungi Recognition: Deep Learning Meets Mycology},
  author={Picek, Lukas and Sulc, Milan and Matas, Jiri and Heilmann-Clausen, Jacob and Jeppesen, Thomas S and Lind, Emil},
  journal={Sensors},
  volume={22},
  number={2},
  pages={633},
  year={2022},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
Output:
{
    "extracted_code": "import timm\nimport torch\nimport torchvision.transforms as T\nfrom PIL import Image\nfrom urllib.request import urlopen\nmodel = timm.create_model(\"hf-hub:BVRA/resnet18.in1k_ft_df20_299\", pretrained=True)\nmodel = model.eval()\ntrain_transforms = T.Compose([T.Resize((299, 299)), \n                              T.ToTensor(), \n                              T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) \nimg = Image.open(PATH_TO_YOUR_IMAGE)\noutput = model(train_transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor"
}
