# Insurance Claims Fraud Detection

This repository contains a **fraud detection** project that combines **IsolationForest** (inverted logic) and **XGBoost** to classify fraud types, complemented by a **Fraud-Ring Graph** analysis using **NetworkX**. The ultimate goal is to identify suspicious insurance claims more accurately by leveraging both **supervised** and **unsupervised** methods, as well as graph-based link analysis.

---

## Overview

- **Isolation Forest (Inverted)**: An unsupervised approach that treats “fraud” as the inlier and tries to isolate everything else. By “inverting” the typical usage, we can pinpoint claims that look too “normal” (or deviant from the known-fraud cluster).  
- **XGBoost**: A powerful supervised classifier trained on known labels (fraud vs. non-fraud or different fraud types).  
- **Fraud-Ring Graph**: A hub-and-spoke graph that connects claims sharing attributes such as city, state, occupation, etc., allowing detection of fraud “rings” via connected components and local fraud ratios.

---

## Key Features

1. **Data Preprocessing & Feature Engineering**  
   - Cleansing raw insurance claim data.  
   - Generating relevant features (e.g., binned numeric features, ratio columns like Premium-to-Sum Assured).  
   - One-hot encoding of categorical fields like occupation or nominee relation.

2. **Hybrid Model (IsolationForest + XGBoost)**  
   - **IsolationForest** (inverted logic) to find claims that resemble a known-fraud distribution.  
   - **XGBoost** (softprob) to classify multiple fraud types or detect genuine vs. suspicious claims.

3. **Graph Analysis (Fraud-Ring Detection)**  
   - Construction of a **NetworkX** graph, connecting each claim to “hub” nodes representing shared attributes.  
   - Calculation of local fraud ratios or detection of suspicious clusters.

4. **Streamlit App**  
   - A user-friendly interface to input claim details.  
   - On-the-fly predictions using the trained models and logic.  
   - Visual outputs or textual displays explaining the likely fraud type (if detected).

5. **Random Sampling & Hybrid Prediction**  
   - Scripts to randomly select claims and generate predictions for demonstration.  
   - Optionally integrate the unsupervised anomaly score from IsolationForest with supervised XGBoost outputs to refine final decisions.

---

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

- **`data/`**: place your raw CSV files or intermediate data sets here.  
- **`models/`**: holds the trained model files (`.pkl`).  
- **`scripts/`**: all Python scripts for preprocessing, EDA, training, hybrid inference, etc.  
- **`app.py`**: the Streamlit-based interactive app.  
- **`requirements.txt`**: list of Python dependencies.  
- **`README.md`**: you’re reading it!

---

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
networkx
...
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
  - An **IsolationForest** on the known-fraud subset (using inverted logic).
  - An **XGBoost** classifier on the labeled data (multiple classes or binary fraud vs. genuine).
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
2. **Toggle** extra fields (like occupant, nominee relation) as needed.  
3. Click **“Predict Fraud?”**.  
4. The app will:
   - Build a single-row DataFrame with the required columns.
   - Compute ratio columns if needed (`Premium-to-SumAssured`, etc.).
   - Call the **inverted** IsolationForest to see if it’s likely FRAUD, and if so, pass to XGBoost for final classification (or vice versa).
5. The UI displays either:
   - **“No Fraud Detected”** if the anomaly score is low or the classifier indicates genuine.  
   - **“Fraud Detected”** along with the predicted fraud type or probability chart if suspicious.

---

## How It Works (Detailed Flow)

1. **Load & Clean**  
   - Ingest the raw insurance claim data from `data/`.  
   - Remove duplicates, handle missing values, convert columns to the right data types.

2. **Train XGBoost**  
   - Split the cleaned data into train/test sets.  
   - Fit an XGBoost model to distinguish fraud vs. genuine claims (or multiple fraud types).  
   - Evaluate with metrics like **accuracy**, **precision/recall**, **F1**, and **ROC-AUC**.

3. **Apply Isolation Forest (Inverted)**  
   - Treat known-fraud examples as “inliers,” and everything else as potential outliers.  
   - Identifies top anomalous claims that deviate from the known-fraud pattern.  
   - Helps catch potential new or rare fraud patterns that XGBoost might miss.

4. **Build Fraud-Ring Graph** (Optional Advanced Step)  
   - Construct a **NetworkX** graph.  
   - Each **claim** is connected to “hub” nodes representing shared attributes (city, state, etc.).  
   - By analyzing connected components and local fraud ratios, you can detect suspicious “rings” of claims that share too many identical attributes.

5. **Generate Outputs & Visuals**  
   - The final pipeline outputs predictions for each claim: anomaly scores from IsolationForest, class probabilities from XGBoost, or ring-based suspiciousness.  
   - The **Streamlit** app shows interactive visuals and results.  
   - Graph-based visualizations (if using the fraud-ring approach) highlight subgraphs of closely connected claims.

---

## Requirements

- **Python 3.8+**  
- Common Python data libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
- **Modeling**: `xgboost`, `imbalanced-learn` (optional)  
- **Graph**: `networkx`  
- **UI**: `streamlit`  

Check the [requirements.txt](./requirements.txt) or run:

```bash
pip install -r requirements.txt
```

---

## Common Issues

- **Mismatch in columns**: Ensure the columns in the Streamlit UI match those used for training.  
- **File paths**: If `.pkl` or `.csv` files aren’t found, adjust the paths in the scripts or re-check your `models/` and `data/` directories.  
- **No Non-Fraud**: If your dataset has no genuine claims, the models may always predict fraud. Consider balancing or adding known non-fraud samples.

---

## Contributing

Contributions are welcome! Please open a GitHub issue or submit a pull request. Ensure you follow the existing code style in the `scripts/` directory.  

---

**Enjoy the Hybrid Fraud Detection Project!** If you have any questions or issues, feel free to [open an issue](https://github.com/your-username/my_streamlit_app/issues) or reach out.

```
