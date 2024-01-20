import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from keras.datasets import mnist


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 28*28)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.view(-1, 1, 28, 28)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.reshape(60000, 1, 28, 28).astype(np.float32)/255.0
training_data = torch.tensor(x_train)


model = AutoEncoder()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

smallest_loss = float("inf")
for epoch in range(EPOCHS):
    print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS))
    epoch_loss = 0
    updates = 0

    for i in range(0, len(training_data), BATCH_SIZE):
        batch = training_data[i:i + BATCH_SIZE].to(DEVICE)

        output = model(batch)

        loss = criterion(output, batch)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        updates += 1

        print(
            "update: " + str(updates) + "/" + str(len(training_data)//BATCH_SIZE + 1) + " | " +
            "loss: " + str(round(epoch_loss/((i + 1)*BATCH_SIZE), 5)), end="\r")
    print(
        "update: " + str(updates) + "/" + str(len(training_data)//BATCH_SIZE + 1) + " | " +
        "loss: " + str(round(epoch_loss/((i + 1)*BATCH_SIZE), 5)))

    if epoch_loss < smallest_loss:
        smallest_loss = epoch_loss

        print("saving new checkpoint ...")
        torch.save(model.encoder.state_dict(), "encoder.pt")
        torch.save(model.decoder.state_dict(), "decoder.pt")
    
