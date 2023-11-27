from google.colab import drive
drive.mount('/content/drive')


import pandas as pd
import numpy as np
import torch
from torch import nn
import random
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = np.load('/content/drive/MyDrive/X_train.npy')
X_test = np.load('/content/drive/MyDrive/X_test.npy')
Y_train = pd.read_pickle('/content/drive/MyDrive/y_train.pickle')
Y_test = pd.read_pickle('/content/drive/MyDrive/y_test.pickle')

X_train = np.swapaxes(X_train, 1, 2)
X_test = np.swapaxes(X_test, 1, 2)

rng = np.random.default_rng()


# Spectrogram Encoding
def spectrogramEncoding(input_array, len_encoding):
    output = np.zeros((input_array.shape[0], 36, len_encoding))
    for i in range(input_array.shape[0]):
        # Num_Channels, Time_Length, Encoding
        output[i, :, :] = spectrogram(input_array[i], nperseg=30)[2].T
    return output


def convertLabelEncoding(label, diseases):
    output = np.zeros(len(diseases))
    for i in range(len(diseases)):
        if label == diseases[i]:
            output[i] = 1
            return output
    print("WARNING")


# Specify Hyperparameters
E = 16
diseases = ['HYP', 'MI', 'NORM', 'STTC', 'CD']
num_epochs = 10000
learning_rate = 1e-8
T = 36
batch_size = 24
max_norm = 2.0
num_channels = 12


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_heads = 2
        self.encoder1 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder2 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder3 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder4 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder5 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder6 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder7 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder8 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder9 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder10 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder11 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.encoder12 = nn.Transformer(d_model=E, nhead=self.num_heads, batch_first=True).encoder.to(device)
        self.linear = nn.Linear(12 * E * T, len(diseases))

    def forward(self, x):
        encoder_output_1 = torch.flatten(self.encoder1(x[:, 0]), start_dim=1)
        encoder_output_2 = torch.flatten(self.encoder2(x[:, 1]), start_dim=1)
        encoder_output_3 = torch.flatten(self.encoder3(x[:, 2]), start_dim=1)
        encoder_output_4 = torch.flatten(self.encoder4(x[:, 3]), start_dim=1)
        encoder_output_5 = torch.flatten(self.encoder5(x[:, 4]), start_dim=1)
        encoder_output_6 = torch.flatten(self.encoder6(x[:, 5]), start_dim=1)
        encoder_output_7 = torch.flatten(self.encoder7(x[:, 6]), start_dim=1)
        encoder_output_8 = torch.flatten(self.encoder8(x[:, 7]), start_dim=1)
        encoder_output_9 = torch.flatten(self.encoder9(x[:, 8]), start_dim=1)
        encoder_output_10 = torch.flatten(self.encoder10(x[:, 9]), start_dim=1)
        encoder_output_11 = torch.flatten(self.encoder11(x[:, 10]), start_dim=1)
        encoder_output_12 = torch.flatten(self.encoder12(x[:, 11]), start_dim=1)
        encoder_output = torch.cat((encoder_output_1, encoder_output_2, encoder_output_3, encoder_output_4,
                                    encoder_output_5, encoder_output_6, encoder_output_7, encoder_output_8,
                                    encoder_output_9, encoder_output_10, encoder_output_11, encoder_output_12), axis=-1)
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
                x[j] = spectrogramEncoding(X_train[index], E)
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
                x[j] = spectrogramEncoding(X_test[index], E)
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
