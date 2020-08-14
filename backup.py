import base64
from io import BytesIO
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt


app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                            'SERVER=LAPTOP-UDLSS6OC;'
                            'DATABASE=TestDB;'
                            'Trusted_Connection=yes;')
cursor = conn.cursor()

df = pd.read_sql_query('SELECT * FROM TestDB.dbo.stationaries',conn)


table = pd.pivot_table(df, values='quantity', index=['department'],
                    columns=['item_code','month1'], aggfunc=np.sum, fill_value=0)
table.plot(kind='bar')
# barchart_png = plt.figure()
# plt.show()
figfile = BytesIO()
plt.savefig(figfile, format='png')
# figfile.seek(0)  # rewind to beginning of file

html_graph = base64.b64encode(figfile.getvalue())

# date = df.iloc[:,0]
# df.iloc[:,0] = date.dt.strftime('%Y-%m-%d')  #Convert time datatype to string ploting purpose
# RECENT_PERIOD = 3
# rolling_mean_3 = df.Electricity_Consumed.rolling(window=3).mean().shift(10)
# fig = plt.figure()
# plt.plot(df.index, df.Electricity_Consumed, label='Electricity Consumed')
# plt.plot(df.index, rolling_mean_3, label='3 Months SMA', color='orange')
# plt.legend(loc='upper left')
#
# figfile = BytesIO()
# plt.savefig(figfile, format='png')
# figfile.seek(0)  # rewind to beginning of file
#
# html_graph = base64.b64encode(figfile.getvalue())

@app.route('/')
def home():
    # This function is called to de-serialize a data stream.
    # html_graph_arima = pickle.load(open('model.pkl','rb'))
    # html_graph_arima = table
    return render_template('bar.html', result1=html_graph.decode('utf8'))
    # return render_template('graph.html', result=html_graph.decode('utf8'))
    # return 'success'

# @app.route('/predict',methods=['POST'])
# def predict():
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#
#     output = round(prediction[0], 2)
#
#     # return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))
#
# @app.route('/results',methods=['POST'])
# def results():
#
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)



if __name__ == "__main__":
    app.run(debug=True)