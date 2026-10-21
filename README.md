# 🧠 Next Word Prediction Using RNN, LSTM, GRU and BiLSTM

## 🛠️ Technologies Used

- 🐍 **Python** – Core programming language
- 🧠 **TensorFlow & Keras** – Deep learning model development
- 🔢 **NumPy** – Numerical computations
- 🐼 **Pandas** – Data handling and processing
- 📚 **NLTK** – Natural language processing and corpus resources
- 🌐 **Streamlit** – Interactive web application
- 💬 **NLP** – Text preprocessing and next-word prediction
- 🤖 **Deep Learning** – Sequence modelling using recurrent neural networks

## 📂 Project Structure

```text
Next-Word-Prediction-Using-RNN-LSTM/
│
├── 📁 artifacts/          # Saved models, vocabulary and metrics
├── 📁 web/
│   └── 🐍 app.py          # Streamlit web application
├── 🐍 data.py             # Built-in training sentences
├── 🐍 main.py             # Training and evaluation pipeline
├── 📄 requirements.txt    # Project dependencies
├── 📄 README.md           # Project documentation
└── 📄 corpus.txt          # Optional text corpus
```

## ✨ Features

- 🎯 Next-word prediction from user input
- 🧠 Four models: RNN, LSTM, GRU and BiLSTM
- 📊 Prediction confidence and alternative word suggestions
- ✍️ Text generation using repeated next-word predictions
- ⚖️ Side-by-side model comparison
- 📈 Training results and evaluation metrics
- 💾 Saved model artifacts for inference
- 🔄 Model retraining support

## 📊 Training Results

The latest completed training run used 10,206 sentences, a vocabulary of 6,966 words, 123,496 training samples and 13,721 validation samples.

| Model | Training Accuracy | Validation Accuracy | Top-3 Accuracy | Perplexity | Epochs |
|---|---:|---:|---:|---:|---:|
| RNN | 16% | 14% | 25% | 252.5 | 32 |
| LSTM | 18% | 15% | 28% | 213.4 | 33 |
| GRU | 17% | 15% | 27% | 210.4 | 32 |
| BiLSTM | 17% | 15% | 27% | 212.5 | 32 |

- 🏆 **Best validation accuracy:** LSTM and GRU/BiLSTM at approximately 15%.
- 📌 **Best top-3 accuracy:** LSTM at 28%.
- 📉 **Lowest perplexity:** GRU at 210.4.

*Note: These are results from one training run. Prediction performance remains limited, so further improvements are needed.*

## 🚀 Future Enhancements

- 📚 Improve dataset quality and vocabulary coverage.
- ⚙️ Optimize model hyperparameters.
- 📊 Evaluate models using a separate test dataset.
- ⚡ Reduce training time through optimization and hardware acceleration.
- ✍️ Improve text preprocessing and generated text quality.
- 🤖 Explore Transformer-based architectures.
- 📈 Add advanced visualizations for model comparison.
- ☁️ Deploy the Streamlit application online.

## ⚠️ Limitations

- 🎯 Current validation accuracy is limited to approximately 14–15%.
- 📚 The dataset may not represent all writing styles and topics.
- 🧩 The models have limited contextual understanding.
- 🔁 Repeated predictions may produce repetitive or grammatically incorrect text.
- ⏳ Training all four models can take several hours.

## 🎓 Learning Outcomes

- Understanding recurrent neural network architectures.
- Text preprocessing and sequence generation.
- Training and evaluating deep learning models.
- Saving and loading model artifacts.
- Building interactive applications with Streamlit.
- Comparing model performance and identifying limitations.

## 👨‍💻 Author

**Dasaratha Sarma**

🔗 GitHub: [BhSarma](https://github.com/BhSarma)

📌 Project Repository: [Next Word Prediction Using RNN and LSTM](https://github.com/BhSarma/Next-Word-Prediction-Using-RNN-LSTM)
