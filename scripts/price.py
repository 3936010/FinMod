import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

SAVE_MODEL = False


data = yf.download('AAPL', start="2010-01-01")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

data['MA_for_250_days'] = data['Close'].rolling(250).mean()
data['MA_for_100_days'] = data['Close'].rolling(100).mean()
data['percentage_change_cp'] = data['Close'].pct_change()
# Adj_close_price = data[['Close']]
features = data[['Close',
                 'MA_for_250_days',
                 'MA_for_100_days',
                 'percentage_change_cp']].dropna()


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(features)
scaled_data

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)
splitting_len = int(len(x_data)*0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)  # shape: (N, seq_len, 1)
        self.y = torch.tensor(y, dtype=torch.float32)  # shape: (N, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Ensure shapes are correct
x_train = x_train.reshape(-1, x_train.shape[1], 1)
x_test = x_test.reshape(-1, x_test.shape[1], 1)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Create datasets
train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size=1, hidden_size1=64, hidden_size2=32, dense_size=32, output_size=1):
        super(LSTMRegressionModel, self).__init__()

        self.lstm1 = nn.GRU(input_size=input_size, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.GRU(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)

        self.fc1 = nn.Linear(hidden_size2, dense_size)
        self.fc2 = nn.Linear(dense_size, output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        out, _ = self.lstm1(x)                # out: (batch, seq_len, hidden_size1)
        out, _ = self.lstm2(out)              # out: (batch, seq_len, hidden_size2)
        out = out[:, -1, :]
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = LSTMRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 12

for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        # forward
        output = model(x_batch)
        loss = criterion(output, y_batch)

        # backward
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = []
    for x_batch, y_batch in test_loader:
        output = model(x_batch)
        predictions.extend(output.squeeze().tolist())

inv_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
print(inv_predictions)

inv_y_test = scaler.inverse_transform(y_test)
print(inv_y_test)

rmse = np.sqrt(np.mean( (inv_predictions - inv_y_test)**2))
print(rmse)

if SAVE_MODEL:
    torch.save(model.state_dict(), 'stock_prediction_model_GRU(1).pth')