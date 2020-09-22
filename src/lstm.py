import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Splitting the data into training and validation set
def train_test_split(x, y, train_split):
    x_train = x[:train_split]
    y_train = y[:train_split]
    x_test = x[train_split:]
    y_test = y[train_split:]
    print('Spliting data and label into training and testing sets', x.shape)
    print('  x_train: ', x_train.shape)
    print('  y_train: ', y_train.shape)
    print('  x_test:  ', x_test.shape)
    print('  y_test:  ', y_test.shape, '\n')

    return x_train, y_train, x_test, y_test


# Generator function for creating random batches of training-data
def batch_generator(batch_size, sequence_length):
    while (True):
        # Numpy array for the data
        x_batch = np.zeros(shape=(batch_size, sequence_length, num_inputs), dtype=np.float16)

        # Numpy array for the labels
        y_batch = np.zeros(shape=(batch_size, sequence_length, 1), dtype=np.float16)

        # Fill each batch with random sequences of data
        for i in range(batch_size):
            # This points somewhere into the training-data.
            idx = np.random.randint(len(x_train) - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train[idx:idx+sequence_length]
            y_batch[i] = y_train[idx:idx+sequence_length]
        
        # Iterate through 256 batches per epoch
        yield (x_batch, y_batch)


# Build the RNN model using LSTM
def build_model():
    # Add topology for the NN
    model = Sequential()
    model.add(LSTM(units=512, return_sequences=True, input_shape=(None, num_inputs,)))
    # model.add(LSTM(units=256, return_sequences=True))  # 1 hidden layer
    # model.add(Bidirectional(LSTM(units=512, return_sequences=True), input_shape=(None, num_inputs,)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    return model


# Create callback: early stopping adn reduced learn rate
def create_callbacks():
    # Early stopping when the model stops improving
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

    # Reduce the learning rate once the model approaches plateau
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                            min_lr=1e-4, patience=0, verbose=1)

    return [callback_early_stopping, callback_reduce_lr]


# Plotting the final results
def plot_comparison(x, y, start, time_steps):
    # Select the sequences from the given start-index
    x = x[start:start + time_steps]
    y = y[start:start + time_steps]
    
    # Use trained model to make predictions
    y_pred = model.predict(np.expand_dims(x, axis=0))
        
    # Scale the data back to original scale
    plt.plot(y_scaler.inverse_transform(y), label='true')
    plt.plot(y_scaler.inverse_transform(y_pred[0]), label='pred')
        
    # Specifying the labels
    plt.ylabel('AvgSpeed')
    plt.legend()
    plt.show()



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
    features_considered = ['DayOfWeek','Hour','Total','AvgSpeed',
                                  'SPD','CLG','VSB','TEMP','DEWP','SLP','ALT','STP','PCP01']
    # features_considered = ['DayOfWeek','DayOfYear','Hour','AvgSpeed']
    df = pd.read_csv(file_sb, usecols=features_considered, float_precision='round_trip')
print(df)  # Checking data


# Choose data from 2016-03-15 00:00:00 (row 1778 - 1) to 2016-11-08 23:00:00 (row 7513 - 1) since it's largest chunk of data without gaps
data = df['2016-03-15':'2016-11-09']  # All data is used if commented out
data = df.dropna()  # Dropping blank data


# Predicting the AvgSpeed 24 hours from now
look_back = 1 * 24  # Look back = one day
df_targets = data.shift(-look_back)
print('Data that is shifted 24 hrs in advance')
print(df_targets)


# Convert Pandas to Numpy arrays, x is input and y is label
x = data.values[:-look_back]
y = df_targets['AvgSpeed'].values[:-look_back]
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
train_split = 30000
x_train, y_train, x_test, y_test = train_test_split(x_scaled, y_scaled, train_split)
num_inputs = x_train.shape[1]


# Create batches of data for the model to train
batch_size = 256          # 256 sets of data per batch
sequence_length = 24 * 7  # One week worth of data per training sequence
generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)
x_batch, y_batch = next(generator)
print('x_batch: ', x_batch.shape)
print('y_batch: ', y_batch.shape, '\n')


# Build the model
model = build_model()


# Train the model
validation_data = (np.expand_dims(x_test, axis=0), np.expand_dims(y_test, axis=0))
model.fit(x=generator, epochs=20, steps_per_epoch=100,
          validation_data=validation_data, callbacks=create_callbacks())


# Evaluate the model using test dataset
result = model.evaluate(x=np.expand_dims(x_test, axis=0),
                        y=np.expand_dims(y_test, axis=0))
print("Validation loss on x_test: ", result)


# Plot the predictions
plot_comparison(x_test, y_test, 3480, 336)
