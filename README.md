# 📰 AG News Text Classification with RNN, LSTM & GRU

A deep learning NLP project that explores and compares **Simple RNN, LSTM, and GRU** architectures for classifying news articles into four categories using the **AG News** dataset.

The project uses both **news titles and article body text**, applies text preprocessing and vectorization, and evaluates each recurrent architecture based on training, validation, and test performance.

---

## 🚀 Project Highlights

* 🧠 Compared **Simple RNN, LSTM, and GRU**
* 📰 Classified news into **4 categories**
* 🔤 Used both **title + article text**
* ⚙️ Applied `TextVectorization` with separate vocabularies
* 🔗 Concatenated title and body representations
* 🛑 Used **Early Stopping** to reduce overfitting
* 📊 Compared **Accuracy and Loss** across datasets
* 💾 Saved the final LSTM model for later use

---

## 📌 News Categories

The AG News dataset contains four classes:

| Category        | Description                   |
| --------------- | ----------------------------- |
| 🌍 **World**    | International and global news |
| ⚽ **Sports**    | Sports-related news           |
| 💼 **Business** | Business and financial news   |
| 🔬 **Sci/Tech** | Science and technology news   |

---

## 📂 Dataset

**Source:** [AG News Dataset — CharCnn_Keras](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)

### Dataset Details

* **Classes:** 4
* **Inputs:** News title + article body
* **Splits:** Train / Validation / Test
* **Validation:** 20% of the original training set
* **Class distribution:** Balanced
* **Missing values:** None
* **Duplicates:** None

---

## 🔄 Project Pipeline

```text
                 AG News Dataset
                        │
                        ▼
               ┌─────────────────┐
               │ Data Loading     │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │      EDA        │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Text Cleaning   │
               └────────┬────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        News Title           Article Body
              │                   │
              ▼                   ▼
     TextVectorization     TextVectorization
              │                   │
              └─────────┬─────────┘
                        ▼
                  Concatenation
                        │
                        ▼
                    Embedding
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
            RNN       LSTM       GRU
              │         │         │
              └─────────┼─────────┘
                        ▼
                  Classification
                        │
                        ▼
              Accuracy & Loss
```

---

## 🧹 Data Preprocessing

The text preprocessing pipeline includes:

* Lowercasing
* HTML tag removal
* Non-alphanumeric character removal
* Whitespace normalization
* Duplicate checking
* Missing-value checking

---

## 🔤 Text Vectorization

The **title** and **article body** are vectorized separately using TensorFlow/Keras `TextVectorization`.

### Configuration

```text
Vocabulary Size : 20,000
Sequence Length : 200
```

The resulting representations are then concatenated before being passed to the neural network.

---

## 🧠 Model Architectures

### 1. Simple RNN

A stacked recurrent architecture using multiple `SimpleRNN` layers with dropout.

```text
Input
  ↓
Embedding
  ↓
SimpleRNN
  ↓
SimpleRNN
  ↓
SimpleRNN
  ↓
SimpleRNN
  ↓
Dropout
  ↓
Output
```

### 2. LSTM

A lightweight LSTM architecture designed to capture longer-term dependencies in the text.

```text
Input
  ↓
Embedding
  ↓
LSTM (16 units)
  ↓
Dropout
  ↓
Dense / Output
```

### 3. GRU

A GRU-based architecture with 32 recurrent units.

```text
Input
  ↓
Embedding
  ↓
GRU (32 units)
  ↓
Dropout
  ↓
Dense / Output
```

---

## ⚙️ Training Configuration

| Parameter           | Value                            |
| ------------------- | -------------------------------- |
| Optimizer           | Adam                             |
| Loss                | Sparse Categorical Cross-Entropy |
| Maximum Epochs      | 25                               |
| Early Stopping      | Yes                              |
| Patience            | 5                                |
| Embedding Dimension | 300                              |
| Vocabulary Size     | 20,000                           |
| Sequence Length     | 200                              |

---

# 📊 Results

The three architectures were evaluated using **training, validation, and test accuracy/loss**.

| Model         | Train Acc | Val Acc | Test Acc | Train Loss | Val Loss | Test Loss |
| ------------- | --------: | ------: | -------: | ---------: | -------: | --------: |
| 🔹 Simple RNN |    26.93% |  27.01% |   28.12% |     1.3843 |   1.3868 |    1.3826 |
| 🟢 LSTM       |    83.69% |  87.80% |   89.09% |     0.4158 |   0.4581 |    0.3982 |
| 🟣 GRU        |    93.72% |  88.31% |   89.41% |     0.2572 |   0.4708 |    0.3740 |

### 📈 Performance Comparison

```text
Test Accuracy

Simple RNN  ██████                         28.12%
LSTM        ██████████████████             89.09%
GRU         ██████████████████             89.41%
```

The **LSTM and GRU models achieved very similar test accuracy**. The GRU achieved a slightly higher test accuracy, while its training accuracy was considerably higher than its validation accuracy, indicating a larger train/validation gap.

For this project, **LSTM was selected as the final model** because it provided a strong validation/test performance with a smaller train/validation gap.

---

## 💾 Model Output

After running the notebook, the selected model is saved as:

```text
best_model.pickle
```

This allows the trained model to be reused without retraining from scratch.

---

## 🛠️ Technologies Used

| Technology                | Purpose                           |
| ------------------------- | --------------------------------- |
| 🐍 **Python**             | Programming language              |
| 🧠 **TensorFlow / Keras** | Deep learning & RNN architectures |
| 🐼 **Pandas**             | Data manipulation                 |
| 📊 **Matplotlib**         | Visualization                     |
| 📈 **Seaborn**            | Statistical visualization         |
| 🔢 **Scikit-learn**       | Data splitting & evaluation       |
| 💾 **Joblib**             | Model serialization               |
| 🌐 **Requests**           | Dataset downloading               |

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/bassam519/ag-news-text-classification-rnn-lstm-gru.git
```

Navigate to the project:

```bash
cd ag-news-text-classification-rnn-lstm-gru
```

Install the dependencies:

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn joblib requests
```

---

## ▶️ Usage

1. Open:

```text
AG_News_Text_Classification.ipynb
```

2. Run the notebook using **Jupyter Notebook** or **Google Colab**.

3. The notebook automatically downloads the AG News dataset.

4. Run the cells sequentially to reproduce:

```text
Data Loading
     ↓
EDA
     ↓
Text Cleaning
     ↓
Text Vectorization
     ↓
Model Training
     ↓
Evaluation
     ↓
Model Saving
```

5. The trained LSTM model will be saved as:

```text
best_model.pickle
```

---

## 📁 Project Structure

```text
ag-news-text-classification-rnn-lstm-gru/
│
├── 📓 AG_News_Text_Classification.ipynb
│
├── 💾 best_model.pickle
│
└── 📄 README.md
```

> `best_model.pickle` is generated after running the notebook.

---

## 🎯 Learning Objectives

Through this project, the following concepts were explored:

* Natural Language Processing (NLP)
* Text preprocessing
* Text vectorization
* Word embeddings
* Recurrent Neural Networks
* LSTM networks
* GRU networks
* Overfitting and generalization
* Early stopping
* Model evaluation
* Multi-class text classification

---

## 📚 References

* [AG News Dataset — CharCnn_Keras](https://github.com/mhjabreel/CharCnn_Keras)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)

---

## 👨‍💻 Author

**Bassam Omar**

Computer Science & AI Student
Interested in **Data Science, Machine Learning, and Deep Learning**.

---

## ⭐ If you find this project useful

Feel free to explore the notebook, experiment with different architectures, and compare the effect of different preprocessing and model configurations.
