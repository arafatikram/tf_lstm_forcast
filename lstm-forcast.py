import numpy as np
import pandas as pd
import plotly.graph_objects as go

#readind datasets
df = pd.read_csv('history_data.csv') #reading file and pre formatting data
df['Date time'] = pd.to_datetime(df['Date time'])
df.set_axis(df['Date time'], inplace=True)
df.drop(['Name', 'Heat Index', 'Snow', 'Snow Depth', 'Wind Gust', 'Wind Chill', 'Conditions'], inplace = True, axis=1)

#plotting datasets
plot1 = go.Scatter(x = df.index, y = df['Temperature'].values, mode = 'lines', name = 'Data')
layout = go.Layout(title = "Datasets",xaxis = {'title' : "Date Time"}, yaxis = {'title' : "magnitude"})
fig = go.Figure(data=[plot1], layout=layout)
fig.show()

#extract main value from datasets
df2  = df['Temperature'].values #extracting datasets from index
df2 = df2.reshape((-1,1))

#datasets splitting
split_percent = 0.80
split = int(split_percent*len(df2))
x_train = df2[:split] # Y axis data
x_test = df2[split:]  # Y axis data
y_train = df['Date time'][:split]  # Y axis data
y_test = df['Date time'][split:]  # Y axis data


#Preparing datasets for model
from keras.preprocessing.sequence import TimeseriesGenerator
look_back = 15
train_generator=TimeseriesGenerator(x_train, x_train, length=look_back, batch_size=20)     
test_generator=TimeseriesGenerator(x_test, x_test, length=look_back, batch_size=1)

#tensorflow-lstm model
from keras.models import Sequential
from keras.layers import LSTM, Dense
model = Sequential()
model.add(
    LSTM(100,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#running model
num_epochs = 30
model.fit(train_generator, epochs=num_epochs, verbose=0)

#predictions
prediction = model.predict(test_generator)

x_train = x_train.reshape((-1))
x_test = x_test.reshape((-1))
prediction = prediction.reshape((-1))

#plotting output of our model
trace1 = go.Scatter(
    x = y_train,
    y = x_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = y_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = y_test,
    y = x_test,
    mode='lines',
    name = 'Actual Data'
)
layout = go.Layout(
    title = "Datasets",
    xaxis = {'title' : "date time"},
    yaxis = {'title' : "magnitude"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()


#function for future prediction
df2 = df2.reshape((-1))

def predict(num_prediction, model):
    prediction_list = df2[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df['Date time'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 60
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)


#complete data plot | forcast included
#forcasting future data
trace1 = go.Scatter(
    x = y_train,
    y = x_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = y_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = y_test,
    y = x_test,
    mode='lines',
    name = 'Real data'
)

trace4 = go.Scatter(
    y = forecast,
    x = forecast_dates,
    mode='lines',
    name = 'Future forcasted data'
)
layout = go.Layout(
    title = "Data Set",
    xaxis = {'title' : "Date time"},
    yaxis = {'title' : "magnitude"}
)
fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
fig.show()
