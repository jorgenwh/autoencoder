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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Encoder()
model.load_state_dict(torch.load("encoder.pt"))
model = model.to(DEVICE)

(_, _), (x_test, _) = mnist.load_data()
x_test = x_test.reshape(10000, 1, 28, 28).astype(np.float32)/255.0
data = torch.tensor(x_test)
data = data.to(DEVICE)

y = model(data)
y = y.cpu().detach().numpy()
np.save("encoded.npy", y)