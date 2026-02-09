# Inference and Training Efficiency in Pruned Multilayer Perceptron Networks
This Project will help you to apply 3 different sparsification methods including "Pre-Training", "Post-Training", "SET" on MLPs. To train each model you just need to go the prefered directory and run the `train.py` file to train your model. 

## Installation
```bash
pip install -r requirements.txt
```

## Check GPU
To check if the GPU is available, run the following command:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
or
```bash
python check_gpu.py
```
