# MNIST Classification
In this directory, we will build a simple neural network to classify the MNIST dataset.

## Dataset
The Dataset will be Downloaded Automatically when you run the code.
The folder structure of `dataset` directory is like this:

```
dataset
│ MNIST
|  | raw
|  |  | files
```

## CLI
To train the model, you can use the following command:
```bash
python train.py -E epochs -B batch_size -L learning_rate -D data_dir
```

Where:
- `epochs`: Number of epochs to train the model.
- `batch_size`: Batch size to use during training.
- `learning_rate`: Learning rate to use during training.
- `data_dir`: Path to the directory containing the dataset.