import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam


data = np.loadtxt("../input/robot-kinematics-dataset/robot_inverse_kinematics_dataset.csv", skiprows=1, delimiter=',')
print(data.shape)
print(type(data))

# Load the CSV file

# Split the input and output variables
q = data[:, :6] # Joint angles
xyz = data[:, 6:] # End-effector position

scaler = MinMaxScaler()
q_scaled = scaler.fit_transform(q)
xyz_scaled = scaler.fit_transform(xyz)

# Split data into training and validation sets
q_train, q_val, xyz_train, xyz_val = train_test_split(q_scaled, xyz_scaled, test_size=0.2, random_state=42) #test_size->20% of the data is used for testing
                                                                                                            #random_state=42 -> we get the same train and test sets across different executions

# Print shapes of training and validation sets
print('Training set shapes:')
print('q_train:', q_train.shape)
print('xyz_train:', xyz_train.shape)

print('\nValidation set shapes:')
print('q_val:', q_val.shape)
print('xyz_val:', xyz_val.shape)

# Define the FNN network architecture
model_fnn = Sequential()
model_fnn.add(Dense(64, input_dim=6, activation='relu'))
model_fnn.add(Dense(32, activation='relu'))
model_fnn.add(Dense(16, activation='relu'))
model_fnn.add(Dense(3))

# Compile the model
model_fnn.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history_fnn = model_fnn.fit(q_train, xyz_train, 
                            epochs=100, 
                            batch_size=32, 
                            validation_data=(q_val, xyz_val))

# Evaluate the model
loss_fnn = model_fnn.evaluate(q_val, xyz_val, verbose=0)
print(f'FNN validation loss: {loss_fnn:.4f}')


plt.plot(history_fnn.history['loss'], label='train')
plt.plot(history_fnn.history['val_loss'], label='val')
plt.title('Training and Validation Loss in FNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
