#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install keras')
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
def generate_time_series(n):
    x = np.arange(n)
    y = np.sin(x/10) + np.random.normal(0, 0.1, n)
    return y.reshape(-1, 1)

# Plot the time series data
def plot_series(series):
    plt.plot(series)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data')
    plt.show()

# Create a train-test split
def train_test_split(data, train_size):
    train_size = int(len(data) * train_size)
    train_data, test_data = data[:train_size], data[train_size:]
    return train_data, test_data

# Define evaluation metrics
def evaluate_model(true, predicted):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    return rmse, mae

# Generate synthetic time series data
n_samples = 1000
time_series_data = generate_time_series(n_samples)
plot_series(time_series_data)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
time_series_data_normalized = scaler.fit_transform(time_series_data)

# Split data into train and test sets
train_data, test_data = train_test_split(time_series_data_normalized, train_size=0.8)


# In[ ]:


# Prepare data for LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Define model architecture
n_steps = 10
n_features = 1
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Prepare data for LSTM
X_train, y_train = prepare_data(train_data, n_steps)
X_test, y_test = prepare_data(test_data, n_steps)

# Reshape input for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)


# In[ ]:


# Make predictions
predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)

# Inverse transform the predictions
predicted_train = scaler.inverse_transform(predicted_train)
predicted_test = scaler.inverse_transform(predicted_test)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
train_rmse, train_mae = evaluate_model(y_train_inv, predicted_train)
test_rmse, test_mae = evaluate_model(y_test_inv, predicted_test)

print(f'Train RMSE: {train_rmse:.2f}, Train MAE: {train_mae:.2f}')
print(f'Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}')

