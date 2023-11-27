from google.colab import drive
drive.mount('/content/drive')


import pandas as pd
import numpy as np
import torch
from torch import nn
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = np.load('/content/drive/MyDrive/X_train.npy')
X_test = np.load('/content/drive/MyDrive/X_test.npy')
Y_train = pd.read_pickle('/content/drive/MyDrive/y_train.pickle')
Y_test = pd.read_pickle('/content/drive/MyDrive/y_test.pickle')

X_train = np.swapaxes(X_train, 1, 2)
X_test = np.swapaxes(X_test, 1, 2)


points_per_sample = 200
X_n = X_train
Y_train = Y_train
X_t = X_test
Y_test = Y_test

X_train = np.zeros((X_n.shape[0], X_n.shape[1], points_per_sample))
X_test = np.zeros((X_t.shape[0], X_t.shape[1], points_per_sample))

downsample_factor = int(X_train.shape[2]/points_per_sample)
for i in range(X_train.shape[0]):
    if i % 1000 == 0:
        print("Sample " + str(i) + " out of " + str(X_train.shape[0]))
    for j in range(X_train.shape[1]):
        for k in range(points_per_sample):
            X_train[i][j][k] = X_n[i][j][k*downsample_factor]

for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        for k in range(points_per_sample):
            X_test[i][j][k] = X_t[i][j][k*downsample_factor]

Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()


def convertToEncoding(input_array, len_one_hot):
    output = np.zeros((input_array.shape[0], input_array.shape[1], len_one_hot))
    idx_array = (input_array*len_one_hot/2 + len_one_hot/2).astype(int)
    idx_array[idx_array>len_one_hot-1] = len_one_hot-1
    idx_array[idx_array<0] = 0
    for i in range(input_array.shape[0]):
        output[:, i, idx_array[:, i]] = 1
    return output


def convertLabelEncoding(label, diseases):
    output = np.zeros(len(diseases))
    for i in range(len(diseases)):
        if label == diseases[i]:
            output[i] = 1
            return output
    print("WARNING")


# Specify Hyperparameters
E = 32
diseases = ['HYP', 'MI', 'NORM', 'STTC', 'CD']
num_epochs = 10000
learning_rate = 1e-8
T = points_per_sample
batch_size = 24
max_norm = 2.0
num_channels = 12


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_heads = 2
        self.encoder = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.linear = nn.Linear(12 * E * T, len(diseases))

    def forward(self, x):
        input = torch.flatten(x, start_dim=1, end_dim=2)
        encoder_output = torch.flatten(self.encoder(input), start_dim=1)
        output = self.linear(encoder_output)

        return output


model = Model().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create Lists for Training and Validation Loss/Error to be Plotted
update_number = []
training_loss_list = []
validation_loss_list = []
training_error_list = []
validation_error_list = []

# Training Run
for i in range(num_epochs):
    x = np.zeros((batch_size, num_channels, T, E))
    y = np.zeros((batch_size, len(diseases)))

    # Randomly Select Samples for the Minibatch
    j = 0
    while j < batch_size:
        index = random.randrange(0, X_train.shape[0])
        try:
            if Y_train[index][0] in diseases:
                x[j] = convertToEncoding(X_train[index], E)
                y[j] = convertLabelEncoding(Y_train[index][0], diseases)
                j += 1
        except:
            pass

    # Convert Training Data and Labels to Torch Tensors
    x_train = torch.from_numpy(x).float().to(device)
    y_train = torch.from_numpy(y).float().to(device)

    y_pred = model(x_train)
    train_loss = loss_fn(y_pred, y_train)

    pred_diseases = torch.argmax(y_pred, dim=-1)
    y_train = torch.argmax(y_train, dim=-1)
    num_correct = torch.sum(y_train == pred_diseases).item()

    # Compute Error as Portion of Incorrect Predicted Labels
    train_error = 1 - num_correct / batch_size

    optimizer.zero_grad()
    train_loss.retain_grad()
    train_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

    test_batch_size = 10

    x = np.zeros((test_batch_size, num_channels, T, E))
    y = np.zeros((test_batch_size, len(diseases)))

    # Randomly Select Samples for the Test Batch
    j = 0
    while j < test_batch_size:
        index = random.randrange(0, X_test.shape[0])
        try:
            if Y_test[index][0] in diseases:
                x[j] = convertToEncoding(X_test[index], E)
                y[j] = convertLabelEncoding(Y_test[index][0], diseases)
                j += 1
        except:
            pass

    # Convert Training Data and Labels to Torch Tensors
    x_val = torch.from_numpy(x).float().to(device)
    y_val = torch.from_numpy(y).float().to(device)

    y_pred_val = model(x_val)
    val_loss = loss_fn(y_pred_val, y_val)

    pred_diseases_val = torch.argmax(y_pred_val, dim=-1)
    y_val = torch.argmax(y_val, dim=-1)
    num_correct = torch.sum(pred_diseases_val == y_val).item()

    # Compute Error as Portion of Incorrect Predicted Labels
    val_error = 1 - num_correct / j

    if i % 10 == 0:
        print("Epoch " + str(i) + ": Train Loss = " + str(train_loss.item()) + ", Train Error = " + str(
            train_error) + ", Validation Loss = " + str(val_loss.item()) + ", Validation Error = " + str(val_error))
        print("Pred Diseases Val = " + str(y_pred_val.T))
        print("Pred Diseases Val = " + str(pred_diseases_val))
        print("            Y Val = " + str(y_val))
        print(j)

    # Collect Training and Validation Loss/Error For Plotting
    if i % 1000 == 0:
        update_number.append(i)
        training_loss_list.append(train_loss.item())
        validation_loss_list.append(val_loss.item())
        training_error_list.append(train_error)
        validation_error_list.append(val_error)

torch.save(model.state_dict(), 'saved_attention.pt')

# Plot Training Loss
plt.plot(update_number, training_loss_list)
plt.title('Attention Model Training Loss as a Function of Number of Updates')
plt.xlabel('Number of Updates')
plt.ylabel('Training Loss')
plt.savefig("AttentionTrainingLoss.pdf", format="pdf")
plt.show()

# Plot Validation Loss
plt.plot(update_number, validation_loss_list)
plt.title('Attention Model Validation Loss as a Function of Number of Updates')
plt.xlabel('Number of Updates')
plt.ylabel('Validation Loss')
plt.savefig("AttentionValidationLoss.pdf", format="pdf")
plt.show()

# Plot Training Error
plt.plot(update_number, training_error_list)
plt.title('Attention Model Training Error as a Function of Number of Updates')
plt.xlabel('Number of Updates')
plt.ylabel('Training Error')
plt.savefig("AttentionTrainingError.pdf", format="pdf")
plt.show()

# Plot Validation Error
plt.plot(update_number, validation_error_list)
plt.title('Attention Model Validation Error as a Function of Number of Updates')
plt.xlabel('Number of Updates')
plt.ylabel('Validation Error')
plt.savefig("AttentionValidationError.pdf", format="pdf")
plt.show()
