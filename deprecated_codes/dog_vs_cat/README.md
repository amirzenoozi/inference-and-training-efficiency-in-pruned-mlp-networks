# Dog Vs Cats Model
In this directory, we will build a model to classify images of dogs and cats. We will use a Convolutional Neural Network (CNN) to build this model. The dataset we will use is the Dogs vs. Cats dataset from Kaggle. The dataset contains 25,000 images of dogs and cats.

## Dataset
For a detailed description of the dataset, please refer to the [Kaggle page](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset).
According to this page, the dataset contains 25,000 images of dogs and cats. The images are labeled as either `dog` or `cat`. The images are of different sizes and have different aspect ratios.
you have to put all the images in a folder named `dataset` in this directory like this:

```
dataset
│ cat
|   |--- cat.0.jpg
|   |--- cat.1.jpg
│ dog
|   |--- dog.0.jpg
|   |--- dog.1.jpg
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