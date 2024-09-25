import os
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from Scripts.dbConnection import cnxn, Data_Inserting_Into_DB

# Query to fetch data
query = f'''
with cte as(
select CAST(predDatetime AS DATE) as Datetime, tickerName, Entry2 as ActualEntry, Exit2 AS ActualExit, [High] AS ActualHigh,
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
from simulationPrediction where tickerName = 'MOL' and predDatetime <= '2024-08-26'
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

# Fetch data
df = pd.read_sql(query, cnxn('mkanalyzer'))

# Separate the first row
first_row = df.iloc[0]
df_remaining = df.iloc[1:]

# Drop rows with NaN values
df_remaining = df_remaining.dropna()

# Define the neural network with additional hidden layers
class AdvancedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(AdvancedNN, self).__init__()
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Preprocess the remaining data
X_remaining = df_remaining[['diffEntry', 'diffExit', 'diffHigh']]
y_remaining = df_remaining['TmPL1']

# Apply SMOTE to the remaining data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_remaining, y_remaining)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Convert categorical labels to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_resampled)

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
hidden_sizes = [128, 64, 32]  # Define hidden layers sizes
output_size = len(label_encoder.classes_)
model = AdvancedNN(input_size, hidden_sizes, output_size)

# Calculate class weights
class_counts = torch.bincount(y_train_tensor)
total_samples = len(y_train_tensor)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights = class_weights.to(torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)
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

# Evaluate the model on the test data
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy on Test Data: {accuracy * 100:.2f}%')

# Predict on the entire dataset excluding the first row
df_remaining_scaled = scaler.transform(df_remaining[['diffEntry', 'diffExit', 'diffHigh']])
X_full_tensor = torch.tensor(df_remaining_scaled, dtype=torch.float32)
model.eval()
with torch.no_grad():
    outputs = model(X_full_tensor)
    _, predictions = torch.max(outputs, 1)

# Add predictions to the original DataFrame excluding the first row
df_remaining['Predicted_TmPL'] = label_encoder.inverse_transform(predictions.numpy())

# Include the prediction for the first row
first_row_scaled = scaler.transform(first_row[['diffEntry', 'diffExit', 'diffHigh']].values.reshape(1, -1))
first_row_tensor = torch.tensor(first_row_scaled, dtype=torch.float32)
with torch.no_grad():
    first_row_output = model(first_row_tensor)
    _, first_row_prediction = torch.max(first_row_output, 1)
first_row['Predicted_TmPL'] = label_encoder.inverse_transform(first_row_prediction.numpy())[0]

# Concatenate the first row with the rest of the DataFrame
df_updated = pd.concat([pd.DataFrame([first_row]), df_remaining], ignore_index=True)
print(df_updated)
