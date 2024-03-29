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

# Define and train a simple recurrent neural network
rnn_model = Sequential()
rnn_model.add(SimpleRNN(64, input_shape=(6, 1), activation='relu'))
rnn_model.add(Dense(3))

rnn_model.compile(loss='mean_squared_error', optimizer=Adam())
rnn_history = rnn_model.fit(q_train.reshape(q_train.shape[0], q_train.shape[1], 1),
                            xyz_train, 
                            epochs=100, 
                            batch_size=32, 
                            validation_data=(q_val.reshape(q_val.shape[0], q_val.shape[1], 1), xyz_val))


plt.plot(rnn_history.history['loss'], label='RNN train')
plt.plot(rnn_history.history['val_loss'], label='RNN val')
plt.title('Training and Validation Loss in RNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
