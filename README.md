# AG News Text Classification with RNN, LSTM, and GRU

A text classification project that compares three recurrent architectures — Simple RNN, LSTM, and GRU — for classifying news articles into four categories using the AG News dataset.

## Overview

This project builds and evaluates deep learning models to classify news headlines and article text into one of four categories:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

Both the article **title** and **body text** are used as input, vectorized separately, and concatenated before being fed into the model.

## Dataset

- **Source:** [AG News CSV (CharCnn_Keras repo)](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)
- **Classes:** World, Sports, Business, Sci/Tech
- **Splits:** Train / Validation (80/20 split of the original train set) / Test
- The dataset is balanced across classes with no missing values or duplicates.

## Pipeline

1. **Data Loading** — Downloads `train.csv`, `test.csv`, and `classes.txt` directly from the source repository.
2. **Exploratory Data Analysis** — Checks class distribution, missing values, and duplicates.
3. **Text Cleaning** — Lowercasing, HTML tag removal, non-alphanumeric character stripping, and whitespace normalization.
4. **Text Vectorization** — Separate `TextVectorization` layers for title and text (vocab size 20,000, sequence length 200), concatenated into a single input tensor.
5. **Modeling** — Three architectures built on a shared embedding layer (dim 300):
   - **Simple RNN:** Stacked 4-layer SimpleRNN with dropout.
   - **LSTM:** Single LSTM layer (16 units) with dropout and L2 regularization.
   - **GRU:** Single GRU layer (32 units) with dropout and L2 regularization.
6. **Training** — Adam optimizer, sparse categorical cross-entropy loss, early stopping on validation loss (patience 5), up to 25 epochs.
7. **Evaluation** — Accuracy and loss compared across train/validation/test sets.

## Results

| Model | Train Acc | Val Acc | Test Acc | Train Loss | Val Loss | Test Loss |
|---|---|---|---|---|---|---|
| Simple RNN | 26.93% | 27.01% | 28.12% | 1.3843 | 1.3868 | 1.3826 |
| LSTM | 83.69% | 87.80% | 89.09% | 0.4158 | 0.4581 | 0.3982 |
| GRU | 93.72% | 88.31% | 89.41% | 0.2572 | 0.4708 | 0.3740 |

**Selected model: LSTM.** GRU and LSTM achieve nearly identical test accuracy, but GRU shows a larger train/validation gap (overfitting), while LSTM generalizes more consistently. LSTM was saved as the final deliverable.

## Requirements

- Python 3
- TensorFlow / Keras
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- requests

Install with:

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn joblib requests
```

## Usage

1. Open and run `AG_News_Text_Classification.ipynb` in Jupyter or Google Colab.
2. The notebook automatically downloads the AG News dataset on first run.
3. Run all cells sequentially to reproduce data preprocessing, model training, and evaluation.
4. The best-performing model (LSTM) is saved as `best_model.pickle`.

## Project Structure

```
.
├── AG_News_Text_Classification.ipynb   # Main notebook: data prep, models, training, evaluation
├── best_model.pickle                   # Saved LSTM model (generated after running the notebook)
└── README.md
```

## License

This project is provided for educational purposes. The AG News dataset is sourced from the [CharCnn_Keras repository](https://github.com/mhjabreel/CharCnn_Keras).
