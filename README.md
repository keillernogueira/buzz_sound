# Buzz Project

# Projeto RecFaces

## Dependências

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [SKLearn](https://scikit-learn.org/stable/install.html)
- [librosa](https://librosa.org/doc/latest/index.html)

## Usage

```
python main.py --dataset_path DATASET_PATH
               --output_path OUTPUT_PATH
               [--learning_rate LR]
               [--weight_decay WD]
               [--batch_size BATCH_SIZE]
               [--epoch_num EN]
```
 
1. ```dataset_path``` is the root path to the dataset
    1. Required
1. ```output_path``` is the path to save outcomes (such as images and trained models) of the algorithm
    1. Required
1. ```learning_rate``` is the learning rate to train the model
    1. Optional
    2. Default value: ```0.01```
1. ```weight_decay``` is the weight decay used to prevent overfitting
    1. Optional
    2. Default value: ```0.005```
1. ```batch_size``` is the batch size
    1. Optional
    2. Default value: ```64```
1. ```epoch_num``` is the number of epochs used to train the model
    1. Optional
    2. Default value: ```50```

## Utils

1. Base paper: [Paper](https://www.biorxiv.org/content/10.1101/2020.09.07.285502v2) and 
[Code](https://github.com/emmanueldufourq/GibbonClassifier/)
2. [Mel Spectogram info](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
3. https://www.mdpi.com/2076-3417/8/9/1573
