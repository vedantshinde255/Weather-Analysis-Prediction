# Weather-Analysis-Prediction
# 🌤️ Weather Data Analysis and Prediction – Pune (PCMC), May 2025

This project analyzes historical weather data for Pune (PCMC region) for May 2025 and predicts future temperature trends using basic linear regression.

---

## 📁 Files

- `generate_dataset.py`  
  Creates the `pune_may_2025_temperature.csv` file with daily average, minimum, and maximum temperatures for May 2025.

- `temperature_analysis.py`  
  Loads the dataset, trains a regression model, and predicts the average temperature for the next 30 days. It visualizes:
  - Historical temperature trends
  - Future forecast using linear regression

- `pune_may_2025_temperature.csv`  
  Generated dataset with daily temperatures for May 1–28, 2025.

---

## 📊 Sample Dataset

| Date       | Min Temp (°C) | Max Temp (°C) | Avg Temp (°C) |
|------------|---------------|---------------|---------------|
| 2025-05-01 | 22.5          | 41.2          | 31.9          |
| 2025-05-02 | 21.2          | 40.6          | 30.9          |
| ...        | ...           | ...           | ...           |

---

## 📈 Technologies Used

- Python 3
- Pandas
- Matplotlib
- scikit-learn

