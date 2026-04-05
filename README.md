# Aisehack_Final
## Team

| | |
|---|---|
| **Team Name** | The Alchemists |
| **Institution** | SVKM's NMIMS Mukesh Patel School of Technology and Management |
| **Members** | Tanmay Porwal · Akshat Katiyar · Vivek Patil · Mahak Sharma |

---

# India in the Haze: PM2.5 Pollution Forecasting (AISE Hack Phase 2 - Theme 2)

This repository contains the code and models developed for the **ANRF AISE Hack Phase 2, Theme 2: Pollution Forecasting (IITD)**. The objective of this project is to forecast high-resolution spatial PM2.5 pollution levels across India for 16 future hours, based on 10 past hours of meteorological and emission data.

Through rigorous experimentation with various spatio-temporal architectures, feature engineering, and validation strategies, we developed a highly optimized **ConvGRU with Spatial Attention (STA)** model tailored specifically for heavy-tailed episodic pollution events.

---

##  The Best Model: ConvGRU + STA + Episode-Weighted Loss
**File:** `gruupdated.ipynb` (Phase 2 v6)

Our final and best-performing model relies on a lightweight but highly effective architecture designed to capture both the long-range spatial transport of pollution plumes and the temporal accumulation of PM2.5. 

### Key Innovations
1. **Spatial Attention (STA) Module:** Inspired by CBAM/KSC-ConvLSTM, we introduced a 7x7 spatial attention gate applied to the encoder's pooled features *before* the ConvGRU. This allows the recurrent cells to focus on spatially salient, high-emission regional context right from the first timestep.
2. **Feature Pruning for Physical Relevance:** We pruned the input from 22 down to the **12 highest-impact physical drivers** to reduce noise and memory footprint:
   * *Base:* `cpm25`, `u10`, `v10`, `t2`, `pblh`, `SO2`
   * *Derived:* `wind_speed`, `vent_index`, `time_sin`, `time_cos`, `lat`, `lon`
3. **Episode-Weighted Loss:** Since pollution spikes (episodes) are the most critical to forecast, we used a custom blended loss function:
   * **60% L1 + 40% SMAPE** in real/log-z space. (L1 prevents gradient explosion on the extreme tails of the log distribution).
   * **5x Weight Penalty** applied to pixels where PM2.5 > 77.08 µg/m³ (the 2σ competition episode boundary).
   * **Spatial Pearson Loss** kicks in after a linear warmup to ensure spatial correlation.
4. **Mixed Precision Training (AMP):** Enabled `torch.amp.autocast` and `GradScaler` for faster training and reduced VRAM consumption.

### Best Model Metrics
* **Val RMSE:** 20.17 µg/m³
* **Global SMAPE:** 0.2577
* **Episode SMAPE:** 0.2060
* **Episode Correlation:** 0.9304

---

##  Experiments & Iterative Improvements

To arrive at the final architecture, we conducted extensive experiments exploring different neural operator architectures, validation schemas, and handling of meteorological variables. 

### 1. Robust Normalization + ConvLSTM Baseline
**File:** `capchange (1).ipynb` (Phase 2 v4)
* **Architecture:** ConvLSTM with TemporalSpatialAttention, SEBlocks, and SpatialAttention.
* **Focus:** Addressed data leakage by computing robust normalization stats strictly on the training split and applying log1p natively across all heavy-tailed emission features (without artificial scaling). 
* **Result:** Achieved strong global correlation (0.92+) but struggled slightly with memory constraints and over-parameterization (9.9M parameters).

### 2. Micro U-FNO (Fourier Neural Operator)
**File:** `asmo-v3-99.ipynb`
* **Architecture:** We experimented with a Micro U-FNO architecture, learning continuous functional mappings in the frequency domain to capture global spatial dependencies.
* **Additions:** Real-time hour-of-day sinusoidal embeddings and a derived `emission_accum_ratio` feature.
* **Inference:** Utilized 4-way Test-Time Augmentation (TTA) to boost robustness.
* **Result:** Highly parameter-efficient (~941K params) and fast, but the frequency-domain approach was slightly less effective at localizing sharp, sudden episodic spikes compared to our STA-ConvGRU.

### 3. LOMO (Leave-One-Month-Out) Validation
* To ensure our model wasn't overfitting to specific seasonal transitions, we implemented a rigorous **LOMO validation strategy**.
* We iteratively trained the model on three months and validated it on the fourth, averaging the metrics. This gave us high confidence in the model's generalizability across diverse seasonal weather patterns (e.g., monsoon vs. winter).

### 4. Zero-Shot Winter Generalization (Train 3 Months, Test December)
* December in India represents the peak of the winter pollution crisis (extreme inversion layers, low PBLH, crop burning residue). 
* We strictly trained our model on April, July, and October, using December entirely as a held-out test set.
* This experiment proved that our feature engineering (specifically the `vent_index` and `pblh` interactions) allowed the model to physically deduce winter accumulation behavior without having seen December data during training.

---

## ⚙️ How to Run

1. Ensure your data is structured in the required format under `raw/` and `test_in/` directories as provided by the competition.
2. Environment Requirements:
   * Python 3.10+
   * `torch`, `numpy`, `scipy`, `pandas`
   * CUDA-enabled GPU (NVIDIA T4 / P100 / A100 recommended)
3. Run the best model:
   ```bash
   jupyter nbconvert --to notebook --execute gruupdated.ipynb
