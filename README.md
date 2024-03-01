# point-bidding
!pip install pandas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# 初始化模型
model = LinearRegression()


# 讀取數據集
df = pd.read_excel("TKE3100.xls")

# 將類別變量進行獨熱編碼
df_encoded = pd.get_dummies(df)

# 將數據集分為特徵和目標變量
X = df_encoded.drop(columns=['權重'])
y = df_encoded['權重']

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 值限制在 0 到 130 之間

y_train_clipped = np.clip(y_train, 0, 130)

y_test_clipped = np.clip(y_test, 0, 130)
# 初始化線性回歸模型
model = LinearRegression()

# 將模型擬合到訓練集上
model.fit(X_train_clipped, y_train_clipped)

# 在測試集上進行預測
y_pred = model.predict(X_test_clipped)

# 計算模型的均方誤差
mse = mean_squared_error(y_test_clipped, y_pred)
print("Mean Squared Error:", mse)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)
