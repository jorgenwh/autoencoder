import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Decoder()
model.load_state_dict(torch.load("decoder.pt"))
model = model.to(DEVICE)

data = np.load("encoded.npy")
data = torch.tensor(data)
data = data.to(DEVICE)

y = model(data)
y = y.cpu().detach().numpy()
print(y.shape)
np.save("decoded.npy", y)