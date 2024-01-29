import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# inputs: ref, predict
# outputs:[mae, mse, rmse, mape, r2]
def evalMetrics(y_ref, y_predict):
    mae = mean_absolute_error(y_ref, y_predict)
    mse = mean_squared_error(y_ref, y_predict)
    rmse = np.sqrt(mean_squared_error(y_ref, y_predict))
    mape = mean_absolute_percentage_error(y_ref, y_predict)
    r_2 = r2_score(y_ref, y_predict)
    return [mae, mse, rmse, mape, r_2]


