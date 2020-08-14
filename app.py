import base64
from io import BytesIO
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import calendar

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

app = Flask(__name__)

conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                            'SERVER=LAPTOP-UDLSS6OC;'
                            'DATABASE=TestDB;'
                            'Trusted_Connection=yes;')
cursor = conn.cursor()

@app.route('/<id>/<year>', methods=['GET', 'POST'])
def barChartPlot(id, year):

        # connect to sql server and read all of data to dataframe: df
        df = pd.read_sql_query('SELECT * FROM TestDB.dbo.stationaries', conn)

        # create column "year" and "month", convert year: int to string, month: int to month_name
        df['year'] = pd.DatetimeIndex(df['date']).year
        df['year'] = df['year'].apply(lambda x: str(x))
        df['month'] = pd.DatetimeIndex(df['date']).month
        df['month'] = df['month'].apply(lambda x: calendar.month_name[x])

        # filter dataframe using variables id and year
        df = df[df.item_code == id]
        df = df[df.year == year]

        # create pivot table based on filtered dataframe (specific itemcode and year)
        table = pd.pivot_table(df, values='quantity', index=['item_code'],
                               columns=['month'], aggfunc=np.sum, fill_value=0)
        # plot the bar chart and save figure
        table.plot(kind='bar')
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        # encode the bar chart figure
        html_graph = base64.b64encode(figfile.getvalue())
        return html_graph.decode('utf8')
        # return render_template('bar.html', result1=html_graph.decode('utf8'))

import json
@app.route('/itemcode/<itemcode>',methods=['GET', 'POST'])
def predict(itemcode):

    return jsonify(arima(itemcode))
    # return html_graph_arima.decode('utf8')
    # return render_template('bar.html', result=html_graph_arima.decode('utf8'))

# @app.route('/results',methods=['POST'])
# def results():
#
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)


def arima(itemcode):
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                          'SERVER=LAPTOP-UDLSS6OC;'
                          'DATABASE=TestDB;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    # connect to sql server and read all of data to dataframe: df
    df = pd.read_sql_query('SELECT * FROM TestDB.dbo.stationaries_ML', conn, parse_dates=['date'], index_col=['date'])
    # if itemcode != None:
    #     df = df[df.item_code == itemcode]

    # itemcode = '1'
    # df = df[df.item_code == itemcode]

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

    # decomposition = seasonal_decompose(df_log, freq=36)
    regressor = ARIMA(df_log, order=(3, 0, 1))
    results = regressor.fit(disp=-1)

    predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(df_log['quantity'].iloc[0], index=df_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    # predictions_ARIMA = np.exp(predictions_ARIMA_log)

    # ML forcaste line chart plot
    results.plot_predict(1, 144)

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
    # To be adjust in the future
    cycle = 36
    differenced = difference(X, cycle)
    # fit model
    model = ARIMA(differenced, order=(7, 0, 1))
    model_fit = model.fit(disp=0)
    # one-step out-of sample forecast
    forecast = model_fit.forecast(steps=12)[0]
    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1
    forecast_arr = []
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, cycle)
        # print('Day %d: %f' % (day, inverted))
        forecast_arr.append(inverted)
        day += 1
    forecast_arr = np.concatenate(forecast_arr)
    fig_arima = BytesIO()
    # Assign plt to fig_arima variable
    plt.savefig(fig_arima, format='png')
    fig_arima.seek(0)
    img = base64.b64encode(fig_arima.getvalue())
    # The JSON format only supports unicode strings.
    # Since base64.b64encode encodes bytes to ASCII-only bytes,
    # you can use that codec to decode the data
    img_json = img.decode('ascii')
    # JSON serializable the numpy array
    forecast_arr_json = pd.Series(forecast_arr).to_json(orient="values")
    data_set = {"forecast": forecast_arr_json, "img": img_json}
    return data_set

if __name__ == "__main__":
    app.run(debug=True)