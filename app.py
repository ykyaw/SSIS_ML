from flask_cors import CORS, cross_origin
from flask import Flask, jsonify
import calendar

import pyodbc
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
# test for upload to git
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'

                            'SERVER=DESKTOP-FNNAE5H;'
                            'DATABASE=SSIS;'
                            'Trusted_Connection=yes;')

cursor = conn.cursor()

@app.route("/cors")
@cross_origin()
def list_users():
  return "cors message success"

@app.route('/<ProductId>/<year>', methods=['GET', 'POST'])
@cross_origin()
def barChartPlot(ProductId, year):

        # connect to sql server and read all of data to dataframe: df
        df = pd.read_sql_query('SELECT t1.Id, t1.DepartmentId, t1.CreatedDate, t2.ProductId, t2.QtyNeeded FROM [SSIS].[dbo].[Requisitions] AS t1 LEFT JOIN [SSIS].[dbo].[RequisitionDetails] AS t2 ON t1.Id = t2.RequisitionId',conn)
        df['CreatedDate'] = np.array(df['CreatedDate']).astype('datetime64[ms]')
        print()
        # create column "year" and "month", convert year: int to string, month: int to month_name
        df['year'] = pd.DatetimeIndex(df['CreatedDate']).year
        df['year'] = df['year'].apply(lambda x: str(x))
        df['month'] = pd.DatetimeIndex(df['CreatedDate']).month
        df['month'] = df['month'].apply(lambda x: calendar.month_name[x])

        # filter dataframe using variables id and year
        df = df[df.ProductId == ProductId]
        df = df[df.year == year]

        # create pivot table based on filtered dataframe (specific itemcode and year)
        table = pd.pivot_table(df, values='QtyNeeded', index=['ProductId'],
                               columns=['month'], aggfunc=np.sum, fill_value=0)
        # plot the bar chart and save figure
        column_order = ['February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                        'November', 'December']
        table2 = table.reindex(column_order, axis=1)
        table2.plot(kind='bar')
        plt.xlabel("Month")
        plt.ylabel("Quantity")
        plt.title("Bar plot of Monthly Product Requisition")
        # plt.show()
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        # encode the bar chart figure
        html_graph = base64.b64encode(figfile.getvalue())
        print(html_graph.decode('utf8'))
        return html_graph.decode('utf8')
        # return render_template('bar.html', result1=html_graph.decode('utf8'))

@app.route('/ProductId/<ProductId>', methods=['GET', 'POST'])
@cross_origin()
def predict(ProductId):

    return jsonify(arima(ProductId))

def arima(ProductId):
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'

                          'SERVER=DESKTOP-FNNAE5H;'

                          'DATABASE=SSIS;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    # connect to sql server and read all of data to dataframe: df
    df = pd.read_sql_query('SELECT t1.CreatedDate, t2.ProductId, t2.QtyNeeded FROM [SSIS].[dbo].[Requisitions] AS t1 LEFT JOIN [SSIS].[dbo].[RequisitionDetails] AS t2 ON t1.Id = t2.RequisitionId', conn)
    df.fillna(0, inplace=True)
    df['CreatedDate'] = np.array(df['CreatedDate']).astype('datetime64[ms]')
    df.index = df['CreatedDate']
    df = df[['ProductId', 'QtyNeeded']]
    print()
    # Filter user selected productId
    if ProductId != 0:
        df = df[df.ProductId == ProductId]
    # For example
    # ProductId = 'C001'
    # df = df[df.ProductId == ProductId]

    df = df[['QtyNeeded']]
    df_log = np.log(df)

    def get_stationarity(timeseries):
        # Dickey???Fuller test:
        result = adfuller(timeseries['QtyNeeded'])
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
    regressor = ARIMA(df_log, order=(6, 0, 1))
    results = regressor.fit(disp=-1)

    predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(df_log['QtyNeeded'].iloc[0], index=df_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    # predictions_ARIMA = np.exp(predictions_ARIMA_log)

    # ML forcaste line chart plot
    results.plot_predict(1, 108)

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
    cycle = 12
    differenced = difference(X, cycle)
    # fit model
    model = ARIMA(differenced, order=(12, 0, 1))
    model_fit = model.fit(disp=0)
    # mult-step out-of sample forecast
    forecast = model_fit.forecast(steps=12)[0]
    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1
    forecast_arr = []
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, cycle)
        # print('Day %d: %f' % (day, inverted))
        history.append(inverted)
        day += 1

    forecast_arr = np.concatenate(history)
    forecast_arr = forecast_arr[-12:]
    forecast_arr = np.rint(forecast_arr)
    fig_arima = BytesIO()
    # Assign plt to fig_arima variable
    plt.ylabel("Quantity in Log Scale")
    plt.xlabel("Time period")
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
