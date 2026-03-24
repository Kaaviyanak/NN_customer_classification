# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="951" height="956" alt="image" src="https://github.com/user-attachments/assets/6f9a93b2-bac0-4d75-ab1f-4f5a6c9577f3" />


## DESIGN STEPS:

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.
### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.
### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).


### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.


## PROGRAM

### Name: KAAAVIYAN K
### Register Number: 212224240066

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```



## Dataset Information
<img width="1241" height="237" alt="image" src="https://github.com/user-attachments/assets/63f0b9b6-115b-449f-92d1-c0151a74145b" />


## OUTPUT

<img width="684" height="573" alt="image" src="https://github.com/user-attachments/assets/2171c11d-683c-4259-a1a2-f69ee97d0732" />

### Confusion Matrix


<img width="548" height="247" alt="image" src="https://github.com/user-attachments/assets/fa46feae-5e26-40ed-ae3c-979a5cfc9a9f" />
<img width="367" height="93" alt="image" src="https://github.com/user-attachments/assets/23c89fc3-d08e-488b-a397-631420f00628" />


### Classification Report
<img width="548" height="247" alt="image" src="https://github.com/user-attachments/assets/fa46feae-5e26-40ed-ae3c-979a5cfc9a9f" />




### New Sample Data Prediction
<img width="349" height="94" alt="image" src="https://github.com/user-attachments/assets/b0b5457a-72c4-4619-81b3-b81dc5e04ca1" />


## RESULT

Thus the neural network classification model was successfully developed.
