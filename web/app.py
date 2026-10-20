
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import METRICS_PATH, MODEL_NAMES, Predictor

st.set_page_config(
    page_title="Next Word AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {
    padding: 1.6rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #182848, #4b6cb7);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 {color: white; margin-bottom: 0.4rem;}
.hero p {color: #e8efff; margin-bottom: 0;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25);
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔮 Next Word Prediction AI</h1>
    <p>Explore text prediction with RNN, LSTM, GRU and BiLSTM neural networks.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return Predictor()


try:
    predictor = get_predictor()
except FileNotFoundError:
    st.error("Trained models were not found. Run `python main.py --retrain` first.")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Choose prediction model", MODEL_NAMES, index=1)
    st.metric("Vocabulary size", len(predictor.word_index) - 1)
    st.caption("Enter words from the model's training vocabulary for best results.")
    if st.button("Reload trained models"):
        get_predictor.clear()
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs([
    "✍️ Predict", "✨ Generate", "⚖️ Compare models", "📊 Training results"
])

with tab1:
    st.subheader("Predict the next word")
    st.write("Enter a phrase and the model will suggest likely next words.")
    user_text = st.text_input(
        "Your phrase",
        placeholder="Example: Python is",
        key="predict_text",
    )

    if user_text.strip():
        results = predictor.predict(user_text, model_name, top_k=5)
        if not results:
            st.warning("No recognized words found. Try a phrase from the known vocabulary.")
        else:
            word, probability = results[0]
            st.success(f"Suggested next word: **{word}**")
            st.metric("Model confidence", f"{probability:.1%}")
            st.markdown("**Other possible words**")
            for candidate, prob in results:
                st.write(f"**{candidate}** — {prob:.1%}")
                st.progress(min(max(prob, 0.0), 1.0))
    else:
        st.info("Try entering: `Python is`, `Deep learning`, or `I love`.")

with tab2:
    st.subheader("Generate a sentence")
    st.write("Start with a few words and let the selected model continue the text.")
    start_text = st.text_input(
        "Starting words",
        placeholder="Example: I love",
        key="gen_text",
    )
    col1, col2 = st.columns(2)
    with col1:
        n_words = st.slider("Additional words", 1, 15, 8)
    with col2:
        temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)

    if st.button("✨ Generate text", type="primary"):
        if not start_text.strip():
            st.warning("Please enter starting words first.")
        else:
            output = predictor.generate(
                start_text, model_name, n_words, temperature
            )
            if output:
                st.markdown("**Generated text**")
                st.info(output)
            else:
                st.warning("No recognized words found. Try different starting words.")
    st.caption("Lower creativity is more predictable; higher creativity adds randomness.")

with tab3:
    st.subheader("Compare the four models")
    st.write("Use the same input to compare each model's top predictions.")
    compare_text = st.text_input(
        "Phrase to compare",
        placeholder="Example: Deep learning",
        key="cmp_text",
    )

    if compare_text.strip():
        rows = []
        for name in MODEL_NAMES:
            predictions = predictor.predict(compare_text, name, top_k=3)
            if predictions:
                rows.append({
                    "Model": name,
                    "Top prediction": predictions[0][0],
                    "Confidence": f"{predictions[0][1]:.1%}",
                    "Top 3 predictions": ", ".join(
                        f"{word} ({prob:.1%})" for word, prob in predictions
                    ),
                })
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("No recognized words found in the input.")

with tab4:
    st.subheader("Training performance")
    if not os.path.exists(METRICS_PATH):
        st.info("No training metrics found. Run `python main.py --retrain`.")
    else:
        with open(METRICS_PATH, encoding="utf-8") as file:
            metrics = json.load(file)

        summary = pd.DataFrame([
            {
                "Model": name,
                "Training accuracy": f"{m['train_acc']:.1%}",
                "Validation accuracy": f"{m['val_acc']:.1%}",
                "Top-3 accuracy": f"{m['val_top3']:.1%}",
                "Perplexity": round(m["perplexity"], 1),
                "Epochs": m["epochs_run"],
                "Training time (s)": m["seconds"],
            }
            for name, m in metrics.items()
        ])

        st.dataframe(summary, hide_index=True, use_container_width=True)
        best_model = max(metrics, key=lambda name: metrics[name]["val_acc"])
        st.metric(
            "Highest validation accuracy",
            f"{metrics[best_model]['val_acc']:.1%}",
            help="This is validation performance, not a guarantee of real-world accuracy.",
        )
        st.caption(
            f"Best validation accuracy in this run: {best_model}. "
            "Lower perplexity is generally better."
        )

        def make_curve(key):
            return pd.DataFrame({
                name: pd.Series(values["history"].get(key, []))
                for name, values in metrics.items()
            })

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Validation loss**")
            st.line_chart(make_curve("val_loss"))
        with col2:
            st.markdown("**Validation accuracy**")
            st.line_chart(make_curve("val_accuracy"))

with st.expander("📚 Explore known vocabulary"):
    words = sorted(
        word for word in predictor.word_index if word != "<unk>"
    )
    st.caption(f"{len(words)} known words")
    st.write(", ".join(words))
