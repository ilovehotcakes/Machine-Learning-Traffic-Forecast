import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR


# Reading file and adjusting file
file_nb = 'processed_data/all_data_sb.csv'  # Southbound traffic
file_sb = 'processed_data/all_data_nb.csv'  # Northbound traffic


# if True: use all of the attributes as inputs, including the speed bins
if (True):
    df = pd.read_csv(file_sb, float_precision='round_trip')  # Use all input
    df['Date'] = df.iloc[:, 0]          # Since file has datatime as index, put datatime into 'Date' col
    del df['Unnamed: 0']                # Remove original index col
    df.set_index('Date', inplace=True)  # Set Date as the index
else:
    features_considered = ['DayOfWeek','DayOfYear', 'Hour','AvgSpeed', 
                            'SPD','CLG','VSB','TEMP','DEWP','SLP','ALT','STP','PCP01']
    # features_considered = ['DayOfWeek','DayOfYear','Hour','Total','AvgSpeed']
    df = pd.read_csv(file_sb, usecols=features_considered, float_precision='round_trip')

# Dropping missing data
data = df.dropna()
print(df)  # Checking data


# Convert Pandas to Numpy arrays, x is input and y is label
x = data.values
y = data['AvgSpeed'].values
y = y.reshape(len(y), 1)
print('Converting dataframe to numpy arrays')
print('  ', x.shape)
print('  ', y.shape, '\n')


# Scaling data to between 0 and 1
x_scaler = MinMaxScaler()
x_scaled = x_scaler.fit_transform(x)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)
print('Scaling data')
print('  ', x_scaled.shape)
print('  ', y_scaled.shape, '\n')


# Split training data and test data
test_size = 3384 / 30000
x_train, _, y_train, _ = train_test_split(x_scaled, y_scaled, test_size=test_size, random_state=1)
# Using the same set of data that the LSTM is using to validate
x_test = x_scaled[33504:33840]
y_test = y_scaled[33504:33840]
print('Spliting training and testing data')
print('   x_train.shape:', x_train.shape)
print('   y_train.shape:', y_train.shape)
print('   x_test.shape:', x_test.shape)
print('   y_test.shape:', y_test.shape, '\n')


# Fitting the model and timing the execution
start_time = time.time()  # Start timer
svr = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=.1)  # Radial basis function kernel
model = svr.fit(x_train, y_train.ravel())
training_time = time.time() - start_time  # End timer
print("SVR training time: %.4f s" % training_time)


# Evaluate the mode using 10 fold cv
scores = cross_val_score(svr, x_train, y_train.ravel(), cv=10, scoring='neg_mean_absolute_error')
print("SVR model score:   %.4f" % scores.mean())  # Model score
scores = cross_val_score(svr, x_test, y_test.ravel(), cv=10, scoring='neg_mean_absolute_error')
print("SVR test score:    %.4f" % scores.mean())  # Test score
# print("Val loss in org scale: %.4f" % y_scaler.inverse_transform(
#                         np.array(0.064).reshape(1, 1)))  # Scale val loss back to original scale


# Predict the same set of data that the LSTM is using to validate
pred = model.predict(x_test)
pred = y_scaler.inverse_transform(pred.reshape(len(pred), 1))
y_test = y_scaler.inverse_transform(y_test.reshape(len(pred), 1))


# Plot the comparison
plt.plot(pred, label='pred')
plt.plot(y_test, label='true')
plt.ylabel('AvgSpeed')
plt.legend()
plt.show()
