import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from numpy import savetxt

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


numpy.random.seed(7)
dataframe = read_csv('Data set/using data/monthly_data.csv', engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
#plt.plot(dataset[1:,1], label = 'Rainfall')
#plt.plot(dataset[:,2],label = 'Snowfall')

dataset = dataset[:,1]
dataset = dataset[:,numpy.newaxis]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform  (dataset)

train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
datasetX, datasetY = create_dataset(dataset, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
datasetX = numpy.reshape(datasetX, (datasetX.shape[0], 1, datasetX.shape[1]))

model = Sequential()
model.add(LSTM(32,return_sequences = True, input_shape=(1, look_back))) 
model.add(LSTM(16,return_sequences = True,input_shape=(1, look_back)))
model.add(LSTM(4,input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=150, batch_size=4, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
datasetPredict = model.predict(datasetX)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
datasetPredict = scaler.inverse_transform(datasetPredict)
datasetY = scaler.inverse_transform([datasetY])


#datasetScore = math.sqrt(mean_squared_error(datasetY[0], datasetPredict[:,0]))
#print('Test Score: %.2f RMSE' % (datasetScore))

metrics.r2_score(trainY[0], trainPredict[:,0])
metrics.r2_score(testY[0], testPredict[:,0])

plt.plot(testY[0,:])
plt.plot(testPredict[:,0])

#plt.plot(trainY[0,:])
#plt.plot(trainPredict[:,0])

MIndices = read_csv('Data set/using data/missing indices.csv', engine='python')
MIndices = MIndices.values
MIndices = MIndices.astype('float32')
indices_size = int(len(MIndices))
original_data_in_gap = numpy.empty([indices_size,1],dtype = 'float')
predicted_data_in_gap = numpy.empty([indices_size,1],dtype = 'float')
original_data_in_gap[:,:] = numpy.nan
predicted_data_in_gap[:,:] = numpy.nan

for i in range(indices_size):
    k = int(MIndices[i,0])
    original_data_in_gap[i] = testY[0,k]
    predicted_data_in_gap[i,0] = testPredict[k,0]
    
plt.plot(original_data_in_gap)
plt.plot(predicted_data_in_gap)
metrics.r2_score(original_data_in_gap, predicted_data_in_gap)

Final_score_without_normalisation = math.sqrt(mean_squared_error(original_data_in_gap, predicted_data_in_gap))
print(Final_score_without_normalisation)


original_data_in_gap_scaled = scaler.fit_transform (original_data_in_gap)
predicted_data_in_gap_scaled = scaler.fit_transform (predicted_data_in_gap)
Final_score_with_normalisation = math.sqrt(mean_squared_error(original_data_in_gap_scaled, predicted_data_in_gap_scaled))
print(Final_score_with_normalisation)

