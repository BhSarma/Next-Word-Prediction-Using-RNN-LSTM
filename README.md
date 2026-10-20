# Next Word Prediction Using RNN and LSTM

Predicts the next word of a sentence using two models (SimpleRNN and LSTM),
with a Streamlit website interface.

## Structure
```
next_word_prediction_rnn_lstm/
├── main.py            # data prep, training, saving, prediction
├── requirements.txt
├── README.md
└── web/
    └── app.py         # Streamlit website
```

## Setup
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## Run
```
python main.py                       # trains both models, prints sample predictions
python main.py --predict "Python is" # quick prediction
python main.py --retrain             # retrain after editing the sentences
streamlit run web/app.py             # open the website
```
