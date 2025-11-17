import os
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from Scripts.dbConnection import cnxn

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load data from database
query = '''
with cte as(
select CAST(predDatetime AS DATE) as Datetime, tickerName,  Entry2 as ActualEntry, Exit2 AS ActualExit, [High] AS ActualHigh,
predTmEntry2 AS PredExtry, predTmExit2 AS PredExit, [Close],
CASE 
    WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
    THEN 1 
    ELSE 0 
END AS TmPL,
CASE 
    WHEN (Entry2 <= predTmEntry2) OR (predTmEntry2 >= [Close])
    THEN 1 
    ELSE 0 
END AS gotEntry,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND (Exit2 >= predTmExit2 OR predTmEntry2 < [Close])) 
    THEN 0
    WHEN (Entry2 <= predTmEntry2 AND predTmEntry2 >= [Close]) OR (predTmEntry2 >= [Close])
    THEN 1 
    ELSE 0 
END AS gotLoss,
CASE 
    WHEN [High] >= predTmExit2 
    THEN 1
    ELSE 0 
END AS gotSell,
CASE 
    WHEN (Entry2 <= predTmEntry2 AND Exit2 >= predTmExit2) 
    THEN ROUND(((predTmExit2 - predTmEntry2) / predTmExit2) * 100, 2)
    ELSE ROUND((([Close] - predTmEntry2) / [Close]) * 100, 2)
END AS ActualProfit, EtEx2Profit as PredProfit
from simulationPrediction where tickerName = 'HINDOILEXP' and predDatetime <= '2024-08-26'
)
select *,  
ROUND((ActualEntry-PredExtry)/PredExtry *100, 2) AS diffEntry,
ROUND((ActualExit-PredExit)/PredExit*100, 2) AS diffExit,
ROUND((ActualHigh-PredExit)/PredExit*100, 2) AS diffHigh,
LAG(CASE WHEN gotSell = 1 THEN ROUND(([Close]-PredExit)/PredExit*100, 2) ELSE 0 END, 1) OVER (ORDER BY Datetime DESC) AS diffClose,
LAG(gotSell, 1) OVER (ORDER BY Datetime DESC) AS sell,
LAG(TmPL, 1) OVER (ORDER BY Datetime DESC) AS TmPL1,
LAG(
    CASE 
        WHEN (TmPL = 1 AND ActualProfit = PredProfit) THEN 'ETEX' 
        WHEN (TmPL = 1 AND ActualProfit != PredProfit) THEN 'ETCL' 
        WHEN (gotEntry = 1 AND gotLoss = 1) THEN 'ETLS'
        ELSE 'NOET' 
    END, 1
) OVER (ORDER BY Datetime DESC) AS TmPred
from cte
order by Datetime DESC
'''
df = pd.read_sql(query, cnxn('mkanalyzer'))
df1 = df.dropna()

# Preprocess the data
X = df1[['diffEntry', 'diffExit', 'diffHigh']]
y = df1['TmPL1']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert categorical labels to numeric values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for training
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(label_encoder.classes_)
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

# Predict on the test data
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# Calculate accuracy
accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Predict on the entire dataset
model.eval()
with torch.no_grad():
    X_full_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    outputs = model(X_full_tensor)
    _, predicted = torch.max(outputs, 1)
    # accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    # print(f'Accuracy: {accuracy * 100:.2f}%')

# Add predictions to the original DataFrame
df1['Predicted_TmPL'] = label_encoder.inverse_transform(predicted.numpy())

# Save the updated DataFrame to a file or database
# df.to_csv('updated_predictions.csv', index=False)
# Alternatively, you can insert the updated DataFrame into a database
# Data_Inserting_Into_DB(df, table_name='predictions_table')

df1 = df1[df1['gotEntry'] == 1]
df1[(df1['TmPL1'] == 1) & (df1['Predicted_TmPL'] == 1)]
