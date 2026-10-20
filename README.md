# 🔮 Next Word Prediction Using RNN, LSTM, GRU and BiLSTM

An interactive deep learning application that predicts the next word in a sentence using four recurrent neural network architectures: RNN, LSTM, GRU and Bidirectional LSTM (BiLSTM).

The project includes next-word prediction, sentence generation, model comparison, and training performance visualization through a Streamlit web interface.

## ✨ Features

- **Next Word Prediction:** Predicts likely next words from user-provided text.
- **Sentence Generation:** Generates text from starting words with a configurable creativity setting.
- **Model Comparison:** Compares predictions from RNN, LSTM, GRU and BiLSTM.
- **Training Results:** Displays training accuracy, validation accuracy, top-3 accuracy, perplexity, epochs and training time.
- **Visualizations:** Shows validation loss and validation accuracy graphs.
- **Interactive Interface:** Provides a user-friendly Streamlit application.

## 🧠 Models Used

| Model | Description |
|---|---|
| RNN | Recurrent Neural Network |
| LSTM | Long Short-Term Memory |
| GRU | Gated Recurrent Unit |
| BiLSTM | Bidirectional Long Short-Term Memory |

## 🛠️ Technologies Used

- Python
- Deep Learning
- Recurrent Neural Networks
- Streamlit
- Pandas
- NumPy
- Matplotlib

## 📂 Project Structure

```text
Next-Word-Prediction-Using-RNN-LSTM/
├── artifacts/       # Saved models and training metrics
├── web/             # Streamlit application
├── data.py          # Training sentences
├── main.py          # Model training and evaluation
├── requirements.txt # Python dependencies
├── README.md        # Project documentation
└── .gitignore       # Files excluded from Git
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/BhSarma/Next-Word-Prediction-Using-RNN-LSTM.git
cd Next-Word-Prediction-Using-RNN-LSTM
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python main.py --retrain
```

This command trains the available models and saves the generated artifacts. Training time depends on your system.

### 5. Run the Streamlit application

If `app.py` is inside the `web` folder, run:

```bash
streamlit run web/app.py
```

If your `app.py` is in the project root, run:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 📊 Model Evaluation

The models are evaluated using:

- **Training Accuracy:** Accuracy on training samples.
- **Validation Accuracy:** Accuracy on held-out validation samples.
- **Top-3 Accuracy:** Whether the correct next word appears among the top three predictions.
- **Perplexity:** A measure of uncertainty in language prediction; lower is generally better.
- **Training Time and Epochs:** Measures of training cost.

Performance depends on the dataset, model configuration and training process. The current dataset is relatively small, so predictions may not always be accurate.

### 📈 Latest Training Results

The following results were obtained from the latest training run using 206 built-in sentences and a vocabulary of 363 words.

| Model | Training Accuracy | Validation Accuracy | Top-3 Accuracy | Perplexity | Epochs |
|---|---:|---:|---:|---:|---:|
| RNN | 54% | 28% | 37% | 98.6 | 37 |
| LSTM | 53% | 27% | 39% | 158.0 | 41 |
| GRU | 31% | 23% | 32% | 122.9 | 34 |
| BiLSTM | 62% | 26% | 34% | 215.7 | 41 |

**Observations:**
- RNN achieved the highest validation accuracy at 28% and the lowest perplexity at 98.6 in this run.
- LSTM achieved the highest top-3 accuracy at 39%.
- BiLSTM achieved the highest training accuracy at 62%, but its validation accuracy was 26%.
- The difference between training and validation results suggests that generalization needs improvement.

*Note: These are results from one training run on a relatively small dataset. They may vary when the models are retrained. Validation accuracy is not a guarantee of performance on unseen real-world text.*

## ⚠️ Limitations

- The training dataset contains a limited number of sentences.
- Predictions may be unreliable for unfamiliar words or sentence patterns.
- Generated sentences may contain repetitive or grammatically incorrect text.
- The reported evaluation metrics should not be interpreted as guaranteed real-world performance.

## 🚀 Future Enhancements

- Expand the dataset with more diverse text.
- Improve model generalization and prediction quality.
- Tune hyperparameters and compare model architectures.
- Add more evaluation examples and testing.
- Deploy the Streamlit application online.

## 👨‍💻 Author

**BhSarma**

GitHub: [@BhSarma](https://github.com/BhSarma)
