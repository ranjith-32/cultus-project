# Advanced Time Series Forecasting with Attention-Based Neural Networks

## Project Overview
This project implements an advanced **attention-based deep learning model** for
**multivariate time series forecasting** using a **Transformer architecture**.
The objective is to move beyond traditional forecasting methods such as ARIMA
or simple LSTMs by leveraging self-attention to capture long-range temporal
dependencies and feature importance.

This project is designed to strictly follow the **Cultus Skills Center – Job
Readiness Project** requirements.

---

## Dataset Description
A **synthetic multivariate time series dataset** was programmatically generated
to ensure full control over complexity and interpretability.

### Dataset Characteristics
- Total time steps: **1,200**
- Number of features: **5**
- Data properties:
  - Multiple seasonal patterns
  - Long-term trend
  - Inter-feature relationships
  - Gaussian noise

### Feature Explanation
| Feature | Description |
|-------|------------|
| F1 | Sinusoidal signal with daily seasonality |
| F2 | Cosine signal with shorter periodic cycle |
| F3 | Linear trend with noise |
| F4 | Interaction between F1 and F2 |
| F5 | High-frequency seasonal component |

Synthetic data generation was chosen to enable **clear attention interpretation**
and controlled experimentation.

---

## Model Architecture
The forecasting model is based on a **Transformer Encoder** implemented in PyTorch.

### Key Components
- Input projection layer
- Multi-head self-attention encoder
- Feed-forward neural network
- Dropout regularization
- Linear output layer

The Transformer architecture was selected for its ability to:
- Capture long-range dependencies
- Learn temporal importance
- Provide interpretability via attention weights

---

## Hyperparameter Tuning
Hyperparameters were manually optimized using **validation loss monitoring**.

| Hyperparameter | Values Tested | Final Choice |
|---------------|--------------|--------------|
| Learning Rate | 0.001, 0.0005 | 0.0005 |
| Sequence Length | 24, 36 | 36 |
| Attention Heads | 2, 4 | 4 |
| Hidden Dimension | 64, 128 | 128 |
| Dropout Rate | 0.1, 0.2 | 0.2 |

The final configuration achieved stable convergence and the lowest validation
error.

---

## Training Strategy
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Batch Size: 64
- Epochs: 20
- Regularization: Dropout

Model training was monitored to ensure stable learning and avoid overfitting.

---

## Baseline Model
A **Naive Forecast Baseline** was implemented:
- Prediction at time *t* equals the value at *t − 1*

This baseline serves as a reference point to quantify the benefits of the
attention-based Transformer model.

---

## Evaluation Metrics
The model was evaluated using **multiple forecasting metrics**, calculated after
inverse scaling to maintain the original data scale:

- **RMSE (Root Mean Squared Error)**
- **MASE (Mean Absolute Scaled Error)**
- **Quantile Loss (τ = 0.5)**

The Transformer model consistently outperformed the baseline across all metrics.

---

## Attention Weight Analysis
The learned attention weights provide meaningful insights:

- High attention is assigned to historical timesteps aligned with known seasonal
  cycles.
- Features representing inter-feature interactions receive higher importance.
- Noisy components receive lower attention, indicating effective noise filtering.

This confirms that the model learns **interpretable and meaningful temporal
dependencies** rather than random correlations.

---

## How to Run the Project
Ensure Python and required libraries are installed, then run:

```bash
python main.py
