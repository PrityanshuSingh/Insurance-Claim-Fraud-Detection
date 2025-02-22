```md
# Insurance Claims Fraud Detection

This repository contains a **fraud detection** project that combines **IsolationForest** (inverted logic) and **XGBoost** to classify fraud types. It includes:

- **Data** directory for CSV files or other data.
- **Models** directory for saving and loading the trained `.pkl` files.
- **Scripts** for data preprocessing, EDA, training, hybrid prediction logic, and random sampling.
- A **Streamlit** app (`app.py`) to provide a user interface for manual input or demonstration.

## Project Structure

```
my_streamlit_app/
├── data/
│   └── ...                       # Your CSVs or other data files
├── models/
│   ├── iso_model.pkl             # Trained IsolationForest (inverted) model
│   ├── xgb_model.pkl             # Trained XGBoost (softprob) model
│   └── label_encoder.pkl         # LabelEncoder for fraud types
├── scripts/
│   ├── preprocessing.py          # Data cleaning, feature engineering, splitting
│   ├── eda.py                    # Exploratory Data Analysis
│   ├── train.py                  # Script to train models and save .pkl
│   ├── hybrid_predict.py         # The inverted hybrid predictor
│   └── random_prediction.py      # Helper code for random sample predictions
├── app.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup Instructions

### 1. Clone or Download the Repository

```bash
git clone https://github.com/your-username/my_streamlit_app.git
cd my_streamlit_app
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Inside the virtual environment (if used), install the packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Your `requirements.txt` might contain lines like:
```
streamlit
xgboost
scikit-learn
pandas
numpy
matplotlib
seaborn
imbalanced-learn
```

### 4. (Optional) Prepare or Update the Data

- Place your dataset(s) in the `data/` folder (e.g. `Augmented_Fraud_Dataset_Final_Updated.csv`).
- Adjust any paths in `scripts/preprocessing.py` or `scripts/train.py` if needed.

### 5. Train the Models (If Not Already Trained)

If you need to train or retrain the models, run:

```bash
python scripts/train.py
```

- This script loads your data (from `data/`), cleans it via `preprocessing.py`, performs EDA (optionally) via `eda.py`, then trains:
  - An **IsolationForest** on the “fraud-only” subset (inverted logic).
  - An **XGBoost** (softprob) classifier on the full multi-class data.
- It then saves the models (`iso_model.pkl`, `xgb_model.pkl`) and the label encoder (`label_encoder.pkl`) into the `models/` directory.

### 6. Run the Streamlit App

After training (or if you already have the `.pkl` files in `models/`):

```bash
streamlit run app.py
```

- This launches a local web server. The terminal will show a URL (usually `http://localhost:8501`).
- Open that URL in your browser. You’ll see the **Fraud Detection App** interface.

### 7. Using the App

1. **Enter** numeric fields (like `ASSURED_AGE`, `POLICY SUMASSURED`, `Premium`, etc.).
2. **(Optional)** Toggle extra fields (like nominee, occupation) if shown, though they might not affect the final model if the model only uses numeric ratio columns.
3. Click **“Predict Fraud?”**.
4. The app will:
   - Build a single-row DataFrame with the required columns.
   - Compute ratio columns if needed (`Premium-to-Sum Assured Ratio`, `Income-to-Sum Assured Ratio`).
   - Call the **inverted** IsolationForest to see if it’s likely FRAUD, and if so, pass to XGBoost for classification.
5. The UI displays “**Fraud Detected**: (Fraud Type)” or “**No Fraud Detected**” plus a probability chart.

### 8. Common Issues

- **Mismatch** between training columns and inference columns:
  - Make sure the columns in `app.py` (or whichever script) exactly match the columns used in `train.py`.
- **File paths**:
  - If the CSV or `.pkl` files aren’t found, adjust the paths in your scripts or place them in the correct folder.
- **Duplicate checkboxes**:
  - If you reuse the same label text in multiple `st.checkbox()`, set a unique `key=` parameter to avoid `StreamlitDuplicateElementId`.

### 9. Contributing

- For bug reports or feature requests, please open an issue or submit a PR.
- Make sure to follow the existing code style.

### 10. License

(Choose an appropriate license, e.g. MIT, Apache, etc.)

---

**Enjoy the Hybrid Fraud Detection App!** If you have any questions, feel free to reach out or open an issue.
```