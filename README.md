# E-Commerce Outlier Mitigation & Feature Engineering

This project focuses on preprocessing highly skewed e-commerce transaction data, handling extreme outliers safely using **Winsorization**, and engineering predictive features to improve machine learning models.

## 🚀 Key Highlights
* **Imputation:** Solved missing categorical data (`CouponCode`) by treating missingness as a behavioral feature (`NO_CODE`).
* **Outlier Mitigation:** Bypassed broken IQR calculations on skewed distributions and implemented a robust **99th Percentile Winsorization** using `numpy.clip()`.
* **Feature Engineering:** Built 3+ highly predictive behavioral, temporal, and category-level features (`Friction_Flag`, `Average_Item_Cost`, `Product_Tier`).

## 🛠️ Tech Stack & Tools
* **Python**
* **Pandas** & **NumPy**
* **SciPy** (for statistical analysis)
* **Seaborn/Matplotlib** (for data visualization)

## 📊 Feature Engineering Highlights

*Average_Item_Cost* - This gives the average price point of the specific items that a customer favours - helps the model distinguish between bulk buyers and luxury buyers. 

*Friction_Flag* - This helps the model determines if a customer will be willing to buy goods again or not using the past orders history. It is predictive because customers who experience order issues are drastically less likely to return. 

*Product_Tier* - This model group products into value tiers based on how their price compares to the rest of the store. 





