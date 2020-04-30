# system imports
import pickle
import numpy as np
from warnings import warn
import random
import json
from datetime import datetime
import os

# pytorch imports
import torch
from torch.nn import L1Loss, MSELoss

# Custom imports
import mol2graph
from scale import normalize
from GraphNet import WeightedGraphNet, MultiGraphNet

assert torch.__version__ == "1.5.0"  # Needed for pytorch-geometric
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    warn("You are using CPU instead of CUDA. The computation will be longer...")


seed = 76583
random.seed(seed)
torch.manual_seed(seed)

# Data parameters
DATASET_TYPE = "medium"
DATA_DIR = f"ala_dipep_{DATASET_TYPE}"
TARGET_FILE = f"free-energy-{DATASET_TYPE}.dat"
# N_SAMPLES = 3815
N_SAMPLES = 21881
NORMALIZE_DATA = False

# Hyperparameters
epochs = 5
criterion = L1Loss()
# criterion = MSELoss()


def get_phi_psi(path: str):
    with open(path, "r") as f:
        m = json.load(f)

    phi, psi = 0, 0
    for dihedral in m["dihedrals"]:
        if [4, 6, 8, 14] == dihedral["atoms"]:
            phi = dihedral["value"]
        if [6, 8, 14, 16] == dihedral["atoms"]:
            psi = dihedral["value"]

    return [phi, psi]


samples = [torch.as_tensor(get_phi_psi(f"{DATA_DIR}/{i}.json")) for i in range(N_SAMPLES)]

train_ind, validation_ind, test_ind = [], [], []

for i in range(0, len(samples), 10):
    # if we are exceeding indexes, put these samples to training
    if i+10 > len(samples):
        train_ind = train_ind + list(range(i, len(samples)))
    else:
        train_ind = train_ind + list(range(i, i+8))
        validation_ind.append(i+8)
        test_ind.append(i+9)

with open(TARGET_FILE, "r") as t:
    target = [torch.tensor([float(v)]) for v in t.readlines()][:N_SAMPLES]

print("Dataset loaded")

model = torch.nn.Sequential(
    torch.nn.Linear(2, 16),
    torch.nn.ReLU6(),
    torch.nn.Linear(16, 1)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                        factor=0.7, patience=5,
#                                                        min_lr=0.00001)
# TODO: print a good summary of the model https://github.com/szagoruyko/pytorchviz
model.train()
print(model)
for i in range(epochs):
    random.shuffle(train_ind)

    losses = []
    for number, j in enumerate(train_ind):
        y_pred = model(samples[j])
        # Compute and print loss
        loss = criterion(y_pred, target[j])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if number % 500 == 0 and number > 0:
            print("{} samples out of {}".format(number, len(train_ind)))

    loss = torch.mean(torch.as_tensor(losses)).item()
    print("Epoch {} - Training loss: {:.2f}".format(i, loss))


predictions = []
errors = []
model.eval()
for j in test_ind:
    prediction = model(samples[j])
    error = prediction - target[j]
    predictions.append(prediction.item())
    errors.append(error.item())

# Compute MAE
mae = np.absolute(np.asarray(errors)).mean()
print("Mean Absolute Error: {:.2f}".format(mae))

# Save predictions as json
directory = f"logs/{DATASET_TYPE}-{datetime.now().strftime('%m%d-%H%M')}"
os.makedirs(directory, exist_ok=True)
with open(f"{directory}/result.json", "w") as f:
    json.dump({
        "predicted": predictions,
        "target": [float(target[i].item()) for i in test_ind],
        "test_frames": test_ind,
    }, f)

torch.save({
    "parameters": model.state_dict()
}, f"{directory}/parameters.pt")

# 1. (To investigate) use dataset batching https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
# 2. Use validation set
# 3. Understand PCA (but should not be used
# 4. Add dropout layer if needed

# 5. check if should use edge_weight or edge_attributes

#############################
# 4. Read paper on NNConv!!!!
# Understand what is graph attention and try pooling methods (top-k?)
# (invert the graph) Very good information is in the edge: find a way to prioritize it and not use the one in the nodes
# Try HypergraphConv layer

# Plus: Understand what pos= on Data means (should not change since I don't use it in the FirstGraphNet)

