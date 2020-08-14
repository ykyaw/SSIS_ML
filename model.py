import pyodbc
import pickle
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import app as app

# stationaries_ML
conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                            'SERVER=LAPTOP-UDLSS6OC;'
                            'DATABASE=TestDB;'
                            'Trusted_Connection=yes;')
cursor = conn.cursor()
# connect to sql server and read all of data to dataframe: df
df = pd.read_sql_query('SELECT * FROM TestDB.dbo.stationaries_ML', conn, parse_dates = ['date'], index_col = ['date'])

df = app.predict.df

# itemcode = '1'
# df = df[df.item_code == itemcode]

# df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.date())
df_col = df.index

# df.set_index('date', inplace=True)
df = df[['quantity']]
df_log = np.log(df)

def get_stationarity(timeseries):
    # Dickey–Fuller test:
    result = adfuller(timeseries['quantity'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift)

df_log.dropna(inplace=True)

decomposition = seasonal_decompose(df_log, freq=36)
regressor = ARIMA(df_log, order=(6,0,1))
results = regressor.fit(disp=-1)

predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(df_log['quantity'].iloc[0], index=df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)

# ML forcaste line chart plot
results.plot_predict(1,144)
#
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

X = df.values
cycle = 36
differenced = difference(X, cycle)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# one-step out-of sample forecast
forecast = model_fit.forecast(steps=12)[0]
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
forecast_arr = []
for yhat in forecast:
	inverted = inverse_difference(history, yhat, cycle)
	print('Day %d: %f' % (day, inverted))
	forecast_arr.append(inverted)
	day += 1
forecast_arr = np.concatenate(forecast_arr)

fig_arima = BytesIO()
# Assign plt to fig_arima variable
plt.savefig(fig_arima, format='png')
fig_arima.seek(0)

html_graph_arima = base64.b64encode(fig_arima.getvalue())
# This function is called to serialize an object hierarchy.
pickle.dump(html_graph_arima, open('model.pkl','wb'))
