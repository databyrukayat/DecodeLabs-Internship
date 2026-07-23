# 🛒 E-Commerce Outlier Mitigation & Feature Engineering
### DecodeLabs Data Science Internship | Python · Pandas · NumPy · Scikit-learn

> **The problem:** Raw e-commerce transaction data is messy;

- extreme outliers break models, missing values introduce bias,
  
- and weak features limit predictive power.
  
> **The solution:** A robust preprocessing and feature engineering pipeline that makes the data model-ready.

---

## 📌 Project Overview

This project tackles one of the most common real-world data science challenges: preparing highly skewed, incomplete transaction data for machine learning without destroying signal in the process.

Working with a real e-commerce dataset during my DecodeLabs internship, I designed a preprocessing pipeline that safely handles extreme outliers, fills missing data intelligently, and engineers three high-signal behavioural features to improve downstream model performance.

---

## 🚀 Key Results

| Challenge | Approach | Outcome |
|-----------|----------|---------|
| Broken IQR on skewed data | 95th Percentile Winsorization via `numpy.clip()` | Outliers capped without data loss |
| Missing `CouponCode` values | Treated missingness as behaviour → `NO_CODE` label | Preserved purchasing pattern signal |
| Weak baseline features | Engineered 3 predictive behavioural features | Stronger model differentiation between customer segments |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)

---

## 🔧 Pipeline Breakdown

### 1️⃣ Outlier Mitigation — Winsorization
Standard IQR-based clipping fails on heavily skewed distributions, it either removes too much or too little. 
Instead, I applied **95th Percentile Winsorization** using `numpy.clip()`:

```python
 lower_bound = data[col].quantile(0.01)
    upper_bound = data[col].quantile(0.95)

data[col] = np.clip(data[col], lower_bound, upper_bound)
```

This caps extreme values without deleting rows, preserving sample size while neutralising outlier distortion.

---

### 2️⃣ Intelligent Imputation — Missing CouponCode
Rather than dropping rows or filling with a generic placeholder, I treated missing coupon codes as a **behavioural signal**:

```python
data['CouponCode'] = data['CouponCode'].fillna('NO_CODE')
```

Customers who don't use coupons behave differently from those who do.
Preserving that distinction gives the model more to work with.

---

### 3️⃣ Feature Engineering — 3 Predictive Features

#### 🔹 `Average_Item_Cost`
```python
data['Average_Item_Cost'] = data['TotalPrice'] / data['ItemsInCart']
```
Distinguishes bulk buyers from luxury buyers, a stronger signal than raw order value alone.

#### 🔹 `Friction_Flag`
```python
data['Order_Failed'] = data['OrderStatus'].isin(['Cancelled', 'Returned', 'Pending']).astype(int)
```
Customers who experienced past order issues are significantly less likely to return. This binary flag makes that pattern explicit and learnable for the model.

#### 🔹 `Product_Tier`
```python
data['Product_Price_Tier'] = pd.qcut(data['UnitPrice'], q=3, labels=['Budget', 'Mid-Range', 'Luxury'])
```
Groups products into value tiers relative to the store's full price range, capturing price sensitivity as a categorical signal.

---

## 📂 Files in This Repo

---

## 💡 Key Learnings

- IQR outlier removal fails silently on skewed data, always visualise distributions first
- Missing data is often a feature, not just noise, context decides which
- Feature engineering adds more model value than hyperparameter tuning in most real datasets
- Clean, commented notebooks are as important as the code itself

---

## 👩🏾‍💻 About Me

**Adeniyi Rukayat** — Data & Operations Analyst | Abuja, Nigeria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/adebimpeadeniyi)
[![Fiverr](https://img.shields.io/badge/Fiverr-1DBF73?style=flat&logo=fiverr&logoColor=white)](https://fiverr.com/ruqqaayyah)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:rukayyahadeniyi@gmail.com)
[![Portfolio](https://img.shields.io/badge/GitHub-databyrukayat-181717?style=flat&logo=github&logoColor=white)](https://github.com/databyrukayat)

---

*DecodeLabs Data Science Internship Program · 2026*




