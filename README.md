# ANN To Graph
This project supposed to extract the adjacency matrix of a graph from an artificial neural network (ANN) model.
The ANN model is trained on a graph dataset and the adjacency matrix of the graph is extracted from the trained model.
The project is implemented in Python using PyTorch.

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

## Usage
```bash
python main.py -P <model_path> -M <model_name> -J <save_as_json> -C <save_as_csv> -V <visualize>
```

where:
- `model_path` is the path to the trained model.
- `model_name` is the name of the model (MNIST / DogVsCat / SMNIST).
- `save_as_json` is a flag to save the adjacency matrix as a JSON file.
- `save_as_csv` is a flag to save the adjacency matrix as a CSV file.
- `visualize` is a flag to visualize the adjacency matrix.