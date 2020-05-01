# TODO: Put the relevant information in nodes
# system imports
import pickle
import numpy as np
from warnings import warn
import random
import json
from datetime import datetime
import os
from pprint import pprint
import tqdm

# pytorch imports
import torch
from torch.nn import L1Loss, MSELoss
from torch_geometric.data import Data, DataLoader

# Custom imports
import mol2graph
from scale import normalize
from GraphNet import WeightedGraphNet, MultiGraphNet, UnweightedDebruijnGraphNet

assert torch.__version__ == "1.5.0"  # Needed for pytorch-geometric
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    warn("You are using CPU instead of CUDA. The computation will be longer...")


# seed = 7653
seed = 76583  # with this it seems magic... maybe I should investigate more...
random.seed(seed)
torch.manual_seed(seed)

# Data parameters
DATASET_TYPE = "big"
DATA_DIR = f"ala_dipep_{DATASET_TYPE}"
TARGET_FILE = f"free-energy-{DATASET_TYPE}.dat"
N_SAMPLES = 3815 if DATASET_TYPE == "small" else 21881 if DATASET_TYPE == "medium" else 50000 if DATASET_TYPE == "old" else 64074 if DATASET_TYPE == "big" else 48952
NORMALIZE_DATA = True
NORMALIZE_TARGET = True
OVERWRITE_PICKLES = False

if not OVERWRITE_PICKLES:
    warn("You are using existing pickles, change this setting if you add features to nodes/edges ")

# Parameters
run_parameters = {
    "graph_type": "De Bruijn",
    "out_channels": 4,
    "convolution": "GraphConv",
    "convolutions": 3,
    "learning_rate": 0.0001 if NORMALIZE_TARGET else 0.001,
    "epochs": 200,
    "normalize_target": NORMALIZE_TARGET,
}

# To check config at the beginning
pprint(run_parameters)

criterion = L1Loss()
# criterion = MSELoss()

graph_samples = []
for i in range(N_SAMPLES):
    try:
        if OVERWRITE_PICKLES:
            raise FileNotFoundError

        with open("{}/{}-dihedrals-graph.pickle".format(DATA_DIR, i), "rb") as p:
            debruijn = pickle.load(p)

    except FileNotFoundError:
        atoms, edges, angles, dihedrals = mol2graph.get_richgraph("{}/{}.json".format(DATA_DIR, i))

        debruijn = mol2graph.get_debruijn_graph(atoms, angles, dihedrals)

        with open("{}/{}-dihedrals-graph.pickle".format(DATA_DIR, i), "wb") as p:
            pickle.dump(debruijn, p)

    graph_samples.append(debruijn)

train_ind, validation_ind, test_ind = [], [], []

for i in range(0, len(graph_samples), 10):
    # if we are exceeding indexes, put these samples to training
    if i+10 > len(graph_samples):
        train_ind = train_ind + list(range(i, len(graph_samples)))
    else:
        train_ind = train_ind + list(range(i, i+3))
        test_ind.append(i+3)
        train_ind = train_ind + list(range(i+4, i+6))
        validation_ind.append(i+6)
        train_ind = train_ind + list(range(i+7, i+9))
        test_ind.append(i+9)

with open(TARGET_FILE, "r") as t:
    target = torch.as_tensor([torch.tensor([float(v)]) for v in t.readlines()][:N_SAMPLES])
    if not NORMALIZE_TARGET:
        target = target.reshape(shape=(len(target), 1))

# Compute STD and MEAN only on training data
target_mean, target_std = 0, 1
if NORMALIZE_TARGET:
    # training_target = torch.tensor([target[i] for i in train_ind])
    target_std = torch.std(target, dim=0)
    target_mean = torch.mean(target, dim=0)
    target = ((target - target_mean) / target_std).reshape(shape=(len(target), 1))

if NORMALIZE_DATA:
    # Single graph normalization
    samples = normalize(graph_samples, train_ind, False)
else:
    samples = graph_samples

dataset = []
for i, sample in enumerate(samples):
    dataset.append(
        Data(x=sample[0], edge_index=sample[1], y=target[i]).to(device)
    )
print("Dataset loaded")


model = UnweightedDebruijnGraphNet(dataset[0], out_channels=run_parameters["out_channels"]).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=run_parameters["learning_rate"])
# optimizer = torch.optim.Adam(model.parameters(), lr=run_parameters["learning_rate"])
# TODO: scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                        factor=0.7, patience=5,
#                                                        min_lr=0.00001)
# TODO: print a good summary of the model https://github.com/szagoruyko/pytorchviz
print(model)
model.train()

for i in range(run_parameters["epochs"]):
    random.shuffle(train_ind)
    for number, j in enumerate(tqdm.tqdm(train_ind)):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(dataset[j].to(device))

        # Compute and print loss
        loss = criterion(y_pred, dataset[j].y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute validation loss
    model.eval()
    val_losses = []
    # save some memory
    with torch.no_grad():
        for j in validation_ind:
            y_pred = model(dataset[j].to(device))
            val_loss = criterion(y_pred, dataset[j].y)
            val_losses.append(val_loss.item())

        val_loss = torch.mean(torch.as_tensor(val_losses)).item()
        if NORMALIZE_TARGET:
            val_loss = val_loss*target_std
        print("Epoch {} - Validation MAE: {:.2f}".format(i+1, val_loss))


predictions = []
errors = []
model.eval()
for j in test_ind:
    # Forward pass: Compute predicted y by passing x to the model
    prediction = model(dataset[j].to(device))
    error = prediction - dataset[j].y
    predictions.append(prediction.item())
    errors.append(error.item())

# Compute MAE
mae = np.absolute(np.asarray(errors)).mean()
if NORMALIZE_TARGET:
    mae *= target_std
print("Mean Absolute Error: {:.2f}".format(mae))

# Save predictions as json
directory = f"logs/{DATASET_TYPE}-{datetime.now().strftime('%m%d-%H%M')}-mae:{mae:.2f}"
os.makedirs(directory)
with open(f"{directory}/result.json", "w") as f:
    json.dump({
        "run_parameters": run_parameters,
        "predicted": predictions,
        "target": [float(target[i].item()) for i in test_ind],
        "target_std": float(target_std),
        "target_mean": float(target_mean),
        "test_frames": test_ind,
    }, f)

torch.save({
    "parameters": model.state_dict()
}, f"{directory}/parameters.pt")


# !!!!. Visualize weights and outputs from layers to see how the NN performs
# !!!!. Visualize FES output of the NN (I have something very close to it) draw a point or histogram2d for each test

# !. Read paper on NNConv after having read slides from geometric deep learning
# !. Understand what is graph attention and try pooling methods (top-k?)
# Try to use batch to avoid loss oscillation
# (....) Try HypergraphConv layer

# ? (To investigate) use dataset batching https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
# ?. Use validation set
# ?. Add dropout layer if needed
# ?. check if should use edge_weight or edge_attributes

# Plus: Understand what pos= on Data means (should not change since I don't use it in the FirstGraphNet)

