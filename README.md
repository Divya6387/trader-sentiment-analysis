# Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science / Analytics Intern Assignment

> **Analyze how Bitcoin market sentiment (Fear/Greed Index) relates to trader behavior and performance on Hyperliquid, and uncover patterns that could inform smarter trading strategies.**
> Data files (`historical_data.csv`, `fear_greed_index.csv`) are not committed. Place them in the root folder before running.
## Setup and How to Run
### Prerequisites
- Python 3.10 or higher
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/primetrade-analysis.git
cd primetrade-analysis
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add your data files
Place both CSVs in the project root:
```
primetrade-analysis/
├── historical_data.csv       ← Hyperliquid trader data
└── fear_greed_index.csv      ← Bitcoin Fear/Greed Index
```

### Step 4A — Run as Jupyter Notebook (recommended)
```bash
jupyter notebook primetrade_analysis.ipynb
```
Then run all cells: **Kernel → Restart and Run All**

### Step 4B — Run as Python script
```bash
python primetrade_analysis.py
```

Charts are saved automatically to the `charts/` folder.

---

## Dataset Overview

| Dataset | Rows | Columns | Key Fields |
|---------|------|---------|------------|
| `historical_data.csv` | 211,224 | 16 | Account, Coin, Execution Price, Size USD, Side, Direction, Closed PnL, Fee, Start Position, Timestamp IST |
| `fear_greed_index.csv` | 2,644 | 4 | date, value (0–100), classification (Extreme Fear to Extreme Greed) |

After merge: 211,218 rows across **479 unique trading days**. Closed-trade events analyzed: **84,820**.

---

## Write-Up Summary

### Methodology

**Data Preparation:**
Both datasets were loaded and audited — zero missing values and zero duplicates were found in either file. Trade timestamps (`Timestamp IST`, format `DD-MM-YYYY HH:MM`) were parsed and aligned to daily granularity. Fear/Greed dates were standardized to `YYYY-MM-DD`. The two datasets were joined via an inner merge on `date`, yielding 479 matched days. Sentiment was simplified from five classes (Extreme Fear, Fear, Neutral, Greed, Extreme Greed) to three (Fear, Neutral, Greed) for binary analysis.

**Feature Engineering:**
Analysis focused on 84,820 closing-direction trades (Close Long, Close Short, Long > Short, Short > Long, Auto-Deleveraging) where PnL is actually realized. Key engineered features: net PnL (`Closed PnL − Fee`), estimated leverage (`Size USD / (|Start Position| × Execution Price)`, clipped 1–100×), win flag (`pnl_net > 0`), and daily long/short ratio. Statistical significance was assessed with a Mann-Whitney U test (non-parametric, appropriate for heavy-tailed PnL distributions).

**Segmentation:**
Three trader segments were built from per-account aggregates: (1) High vs Low Leverage — median split on avg leverage; (2) Frequent vs Infrequent — top-25% by trade count; (3) Consistent Winner vs Loser — quartile split on win rate.

**Predictive Model (Bonus):**
A Gradient Boosting Classifier was trained on daily behavioral features plus 1–3 day lags of win rate, PnL, and Fear/Greed index to predict next-day PnL tertile bucket. 5-fold cross-validation was used to prevent data leakage. Achieved 47.3% accuracy vs 33.3% random baseline.

---

### Key Insights

**Insight 1 — Fear Days Are Significantly More Profitable**

| Metric | Fear Days | Greed Days |
| Win Rate | **85.0%** | 80.0% |
| Mean Net PnL / trade | **$116.77** | $60.17 |
| Avg Loss (drawdown proxy) | –$187.56 | –$175.24 |
| Total Trades | 35,882 | 33,068 |

Mann-Whitney U test confirms the difference is statistically significant (p < 0.0001). Traders on Fear days are more selective — fewer but higher-quality entries — resulting in better win rates and higher average PnL per trade.

**Insight 2 — High Leverage Consistently Destroys Returns**

| Segment | Win Rate | Mean PnL/trade |
| Low Leverage | **88.7%** | **$323** |
| High Leverage | 78.6% | –$43 |

High-leverage traders are profitable on Fear days but severely underperform on Greed days, suggesting they overextend during bullish sentiment and get caught in reversals.

**Insight 3 — Greed Days Drive Herding Into Longs**
Long/Short ratio is **6.5× on Fear days vs 4.3× on Greed days**. On Greed days traders pile into longs (crowded positioning), creating fragility. Average daily trade volume on Fear days (792) is 2.7× higher than on Greed days (294).

**Insight 4 — Behavioral Momentum Beats Sentiment as a Predictor**
In the predictive model, `n_trades` (daily trade count) was the single most important feature (importance: 0.29), followed by lagged PnL. The Fear/Greed index value itself ranked lower than behavioral lag features — *how traders are behaving today* is a stronger near-term signal than *how the market feels today*.

---

### Strategy Recommendations

**Rule 1 — Sentiment-Based Leverage Cap**

> When the Fear/Greed Index drops below 35 (Fear or Extreme Fear), cap leverage at ≤ 3× for all new positions. On Greed days (index > 65), standard leverage is acceptable — but reduce position concentration in long-biased setups to avoid herding risk.

Rationale: High-leverage traders earn mean PnL of –$43 while low-leverage traders earn $323. The penalty for leverage is largest on Greed days when crowding amplifies reversal losses.

**Rule 2 — Sentiment-Aware Trade Frequency**

> Consistent winners (top-quartile win-rate traders) should increase trade frequency on Greed days to capture bullish momentum. Inconsistent frequent traders should reduce frequency on Fear days — overtrading during fearful markets is the primary driver of negative PnL outcomes.

Rationale: Consistent winners maintain ~85%+ win rates across both regimes. Inconsistent traders show their worst PnL per trade on Fear days despite (or because of) trading at 2.7× higher volume.

---

## Output Charts

| Chart | What It Shows |
|-------|---------------|
| `01_performance_fear_greed.png` | Win rate and mean PnL: Fear vs Greed vs Neutral |
| `02_behavior_fear_greed.png` | Trade frequency, position size, L/S ratio, leverage by sentiment |
| `03_segments_heatmap.png` | Win rate heatmap: 3 trader segments × sentiment |
| `04_pnl_distribution.png` | PnL distribution histogram overlay: Fear vs Greed |
| `05_leverage_distribution.png` | Leverage distribution: Fear vs Greed |
| `06_rolling_pnl_vs_sentiment.png` | Rolling 30-day PnL vs Fear/Greed index timeline |
| `07_feature_importance.png` | Gradient Boosting feature importances for next-day prediction |

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.0 | Data loading, cleaning, aggregation |
| numpy | >= 1.24 | Numerical operations |
| matplotlib | >= 3.7 | Charts and visualizations |
| seaborn | >= 0.12 | Heatmaps |
| scipy | >= 1.10 | Mann-Whitney U statistical test |
| scikit-learn | >= 1.3 | Gradient Boosting model, cross-validation |

---

## Submission

**Role:** Data Science / Analytics Intern
**Company:** Primetrade.ai
