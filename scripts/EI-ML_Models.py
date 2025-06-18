import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("./data/real_data.csv")
series = df["Total Cases (Weekly)"].values
train, test = series[:60], series[60:72]

# exogenous (SIBR fit) series
xreg_df = pd.read_csv("./data/SIBR_model_fit.csv")
xreg = xreg_df["Fitted Infected Population"].values
train_xreg, test_xreg = xreg[:60], xreg[60:72]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def mase(y_true, y_pred, y_train):
    d = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / d

def evaluate(test, pred, train, name):
    return {
        "Model": name,
        "MAPE": mape(test, pred),
        "SMAPE": smape(test, pred),
        "MAE": mean_absolute_error(test, pred),
        "MASE": mase(test, pred, train),
        "RMSE": np.sqrt(mean_squared_error(test, pred))
    }

arimax = pm.auto_arima(
    train,
    exogenous=train_xreg.reshape(-1,1),
    seasonal=False,
    stepwise=True,
    suppress_warnings=True
)
pred_arimax = arimax.predict(n_periods=len(test), exogenous=test_xreg.reshape(-1,1))

p = 2
X_train, y_train2 = [], []
for i in range(p, len(train)):
    X_train.append(np.concatenate([train[i-p:i], [train_xreg[i]]]))
    y_train2.append(train[i])
X_train = np.array(X_train)
y_train2 = np.array(y_train2)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

mlp = MLPRegressor(hidden_layer_sizes=(10,12), random_state=100, max_iter=2000)
mlp.fit(X_train_scaled, y_train2)

history = list(train)
y_pred_arnnx = []
for ex in test_xreg:
    x_in = np.concatenate([history[-p:], [ex]])
    x_in_scaled = scaler.transform(x_in.reshape(1, -1))
    yhat = mlp.predict(x_in_scaled)[0]
    y_pred_arnnx.append(yhat)
    history.append(yhat)

perf = []
perf.append(evaluate(test, pred_arimax, train, "EI-ARIMA"))
perf.append(evaluate(test, y_pred_arnnx, train, "EI-ARNN"))
perf_df = pd.DataFrame(perf)
print(perf_df)

pd.Series(pred_arimax, name="EI-ARIMA_Forecast").to_csv("EI-ARIMA_Forecast.csv", index=False)
pd.Series(y_pred_arnnx, name="EI-ARNN_Forecast").to_csv("EI-ARNN_Forecast.csv", index=False)
pd.Series(arimax.predict_in_sample(exogenous=train_xreg.reshape(-1,1)), name="EI-ARIMA_Prediction") \
  .to_csv("EI-ARIMA_Prediction.csv", index=False)
pd.Series(mlp.predict(scaler.transform(X_train)), name="EI-ARNN_Prediction") \
  .to_csv("EI-ARNN_Prediction.csv", index=False)

plt.figure(figsize=(10,5))
plt.plot(range(1,61), train, label="Train", linewidth=1.2)
plt.plot(
    range(1,61),
    arimax.predict_in_sample(exogenous=train_xreg.reshape(-1,1)),
    'o-', label="EI-ARIMA Fit", linewidth=1, markersize=4
)
plt.plot(range(1,61), train_xreg, label="SIBR exog", linewidth=1.2)
plt.xlabel("Time (weeks)"); plt.ylabel("Cases")
plt.title("EI-ARIMA: Training & In-Sample Fit")
plt.legend(); plt.tight_layout()

# EI-ARIMA: Test & Forecast
plt.figure(figsize=(10,5))
plt.plot(range(61,73), test,        label="Test",       linewidth=1.2)
plt.plot(range(61,73), pred_arimax, label="EI-ARIMA FC", linewidth=1)
plt.plot(range(61,73), test_xreg,   label="SIBR exog",  linewidth=1.2)
plt.xlabel("Time (weeks)"); plt.ylabel("Cases")
plt.title("EI-ARIMA: Test & Forecast")
plt.legend(); plt.tight_layout()

# EI-ARNN: Training & In-Sample Fit
arnn_in_sample = mlp.predict(scaler.transform(X_train))
plt.figure(figsize=(10,5))
plt.plot(range(1,61), train,              label="Train",      linewidth=1.2)
plt.plot(range(p+1,61), arnn_in_sample,   'o-', label="EI-ARNN Fit", linewidth=1, markersize=4)
plt.plot(range(1,61), train_xreg,         label="SIBR exog",  linewidth=1.2)
plt.xlabel("Time (weeks)"); plt.ylabel("Cases")
plt.title("EI-ARNN: Training & In-Sample Fit")
plt.legend(); plt.tight_layout()

# EI-ARNN: Test & Forecast
plt.figure(figsize=(10,5))
plt.plot(range(61,73), test,         label="Test",      linewidth=1.2)
plt.plot(range(61,73), y_pred_arnnx, label="EI-ARNN FC", linewidth=1)
plt.plot(range(61,73), test_xreg,    label="SIBR exog", linewidth=1.2)
plt.xlabel("Time (weeks)"); plt.ylabel("Cases")
plt.title("EI-ARNN: Test & Forecast")
plt.legend(); plt.tight_layout()

plt.show()

