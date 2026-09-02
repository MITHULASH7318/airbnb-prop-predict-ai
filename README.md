# PropPredict AI — Airbnb Property Intelligence Platform

> An end-to-end AI platform predicting Airbnb property prices, ROI, and investment signals
> across **Dubai, Goa, and Maldives** using XGBoost ML models + Claude AI chatbot.

---

## Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## What It Does

| Feature | Description |
|---------|-------------|
| Price Predictor | XGBoost model predicts nightly Airbnb price (R² = 0.91) |
| ROI Classifier | Random Forest classifies investment signal: Strong Buy / Buy / Hold / Avoid |
| Investment Scorer | Ranks properties 0–100 based on ROI + location + reviews |
| Occupancy Forecaster | Predicts annual occupancy rate per property |
| AI Chatbot | Claude-powered assistant answers natural language property questions |
| Interactive Dashboard | Streamlit app with maps, charts, and property explorer |

---

## Project Structure

```
airbnb_predictor/
├── data/
│   ├── raw/                   # Dubai, Goa, Maldives CSVs
│   ├── processed/             # Feature-engineered dataset
│   └── generate_dataset.py    # Generates 1500 realistic listings
├── models/
│   ├── train_models.py        # Train all 4 ML models
│   ├── predict.py             # Prediction API used by app + chatbot
│   ├── price_model.pkl        # XGBoost price predictor
│   ├── roi_classifier.pkl     # Random Forest ROI classifier
│   ├── score_model.pkl        # Investment score model
│   └── metrics.json           # Model performance metrics
├── chatbot/
│   └── chatbot.py             # Claude AI chatbot engine
├── utils/
│   └── preprocess.py          # Feature engineering pipeline
├── app/
│   └── app.py                 # Streamlit frontend (5 pages)
├── requirements.txt
├── setup.sh                   # One-command setup
└── README.md
```

---

## Quickstart (Local)

```bash
# 1. Clone
git clone https://github.com/yourusername/airbnb-property-predictor
cd airbnb-property-predictor

# 2. Setup everything (installs deps + generates data + trains models)
bash setup.sh

# 3. Set your Anthropic API key (free tier available)
export ANTHROPIC_API_KEY=your_key_here

# 4. Run the app
streamlit run app/app.py
```

---

## ML Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Price Predictor (XGBoost) | R² Score | 0.91 |
| Price Predictor | MAE | $56 |
| ROI Classifier (Random Forest) | Accuracy | 99.7% |
| Investment Scorer (XGBoost) | MAE | ±2.67 pts |

---

## Deployment — Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file to `app/app.py`
4. Add `ANTHROPIC_API_KEY` in **Secrets** settings
5. Click Deploy — live in 2 minutes

---

## Deployment — Hugging Face Spaces (Free)

```bash
# Install huggingface_hub
pip install huggingface_hub

# Create space and push
huggingface-cli login
huggingface-cli repo create airbnb-predictor --type space --space_sdk streamlit
git remote add hf https://huggingface.co/spaces/yourusername/airbnb-predictor
git push hf main
```

---

## Data Sources

- [Inside Airbnb](https://insideairbnb.com/get-the-data/) — real listing + review data
- [Dubai Pulse / DLD](https://www.dubaipulse.gov.ae) — government property transactions
- [Kaggle Airbnb datasets](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Synthetic data for Maldives/Goa (calibrated to real market distributions)

---

## Business Use Cases

| Market | Problem Solved | Target Client |
|--------|----------------|---------------|
| Dubai | Best-ROI zone for property purchase | Bayut users, DLD investors |
| Goa | Dynamic nightly pricing for villa owners | Independent Airbnb hosts |
| Maldives | Occupancy forecasting for resort revenue | Boutique resort operators |

---

## Tech Stack

`Python` `XGBoost` `Scikit-learn` `Streamlit` `Plotly` `Claude API` `Pandas` `NumPy`

---

## Author

Built as a portfolio project demonstrating end-to-end Data Science + AI/ML skills.
Connect on [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
