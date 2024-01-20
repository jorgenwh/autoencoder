import random
import numpy as np
import matplotlib.pyplot as plt

# display 100 random reconstructed mnist images

decoded_data = np.load("decoded.npy")

fig = plt.figure(figsize=(20, 40))
for i in range(100):
    ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[])
    idx = random.randint(0, len(decoded_data))
    ax.imshow(decoded_data[random.randint(0, len(decoded_data))][0], cmap="gray")

plt.show()
