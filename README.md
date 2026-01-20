# Yahoo-Finance-Analysis-Market-PLTD-Jan-2025--Jan-2026
!pip install yfinance pandas numpy scikit-learn

import yfinance as yf
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.model_selection import train_test_split


import yfinance as yf 
# Choose your ticker
ticker = "PLTD" 
# Download historical data 
data = yf.download(ticker, start="2010-01-01", end="2025-12-31") 
# Display the first rows 
print(data.head()) 
# Save to CSV if needed 
data.to_csv(f"{ticker}_history.csv")

# 2. Create features
data["Return"] = data["Close"].pct_change()
data["MA_10"] = data["Close"].rolling(window=10).mean()
data["MA_5"] = data["Close"].rolling(window=5).mean()
data["Volatility_5"] = data["Return"].rolling(window=5).std()


# Drop initial NaNs
data = data.dropna()

# 3. Create target: 1 if next day close > today close, else 0
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data = data.dropna()

features = ["Return", "MA_5", "MA_10", "Volatility_5"]
X = data[features]
y = data["Target"]

# 4. Train / test split (time-aware: no shuffle)
split_index = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# 5. Train model 
model = RandomForestClassifier( n_estimators=200, max_depth=5,random_state=42 )
model.fit(X_train, y_train)

# 6. Evaluate 
y_pred = model.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Predict for the latest day in the dataset 
latest_features = X.iloc[-1:].values 
prob_up = model.predict_proba(latest_features)[0][1] 
direction = "UP" if prob_up >= 0.5 else "DOWN"

print(f"Predicted next-day direction for {ticker}: {direction}") 
print(f"Probability of going UP: {prob_up:.2f}")

import pandas as pd
import matplotlib.pyplot as plt

# Reset index so Date becomes a column
data = data.reset_index()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data["Date"], data["Close"], label="Close Price", color="blue")
plt.title(f"{ticker} Stock Price History")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.show()



