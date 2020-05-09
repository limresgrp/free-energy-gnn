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
from torch.nn import L1Loss
from torch_geometric.data import Data

# Custom imports
from helpers import mol2graph
from helpers.scale import normalize
from GraphNet import WeightedGraphNet, MultiGraphNet

assert torch.__version__ == "1.5.0"  # Needed for pytorch-geometric
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    warn("You are using CPU instead of CUDA. The computation will be longer...")


seed = 76583
random.seed(seed)
torch.manual_seed(seed)

# Data parameters
DATASET_TYPE = "small"
DATA_DIR = f"ala_dipep_{DATASET_TYPE}"
TARGET_FILE = f"free-energy-{DATASET_TYPE}.dat"
N_SAMPLES = 3815
# N_SAMPLES = 21881
NORMALIZE_DATA = True
OVERWRITE_PICKLES = True

if not OVERWRITE_PICKLES:
    warn("You are using existing pickles, change this setting if you add features to nodes/edges ")

# Parameters
run_parameters = {
    "graph_type": "MultiGraph",
    "out_channels": 8,
    "iterations_normal_graph": 0,
    "iterations": 0,
    "nodes_features": mol2graph.GLOBAL_NODES_FEATURES,
    "edge_features": mol2graph.EDGE_FEATURES,
}

epochs = 10
criterion = L1Loss()
# criterion = MSELoss()

graph_samples = []
for i in range(N_SAMPLES):
    try:
        if OVERWRITE_PICKLES:
            raise FileNotFoundError

        with open("{}/{}.pickle".format(DATA_DIR, i), "rb") as p:
            graph_sample = pickle.load(p)

    except FileNotFoundError:
        atoms, edges, angles, dihedrals = mol2graph.get_richgraph("{}/{}.json".format(DATA_DIR, i))

        interactions_sample = mol2graph.get_atoms_interactions_graph(atoms, edges)
        angles_sample = mol2graph.get_angles_graph(atoms, angles)
        dihedrals_sample = mol2graph.get_dihedrals_graph(atoms, dihedrals, angles)

        graph_sample = [interactions_sample, angles_sample, dihedrals_sample]
        with open("{}/{}.pickle".format(DATA_DIR, i), "wb") as p:
            pickle.dump(graph_sample, p)

    graph_samples.append(graph_sample)

train_ind, validation_ind, test_ind = [], [], []

for i in range(0, len(graph_samples), 10):
    # if we are exceeding indexes, put these samples to training
    if i+10 > len(graph_samples):
        train_ind = train_ind + list(range(i, len(graph_samples)))
    else:
        train_ind = train_ind + list(range(i, i+8))
        validation_ind.append(i+8)
        test_ind.append(i+9)

with open(TARGET_FILE, "r") as t:
    target = torch.as_tensor([torch.tensor([float(v)]) for v in t.readlines()][:N_SAMPLES])

# Compute STD and MEAN only on training data
target_mean, target_std = 0, 1
if NORMALIZE_DATA:
    training_target = torch.tensor([target[i] for i in train_ind])
    target_std = torch.std(training_target, dim=0)
    target_mean = torch.mean(training_target, dim=0)
    target = ((target - target_mean) / target_std).reshape(shape=(len(target), 1))

    columns = [[samples[i] for samples in graph_samples] for i in range(len(graph_samples[0]))]
    normalized_columns = [normalize(column, train_ind) for column in columns]
    graph_samples = [[column[i] for column in normalized_columns] for i in range(len(graph_samples))]


dataset = []
for i, samples in enumerate(graph_samples):
    dataset.append(
        [Data(x=sample[0], edge_index=sample[1], edge_attr=sample[2], y=target[i]).to(device) for sample in samples]
    )
print("Dataset loaded")


interactions_model = WeightedGraphNet(dataset[0][0], iterations=run_parameters["iterations_normal_graph"], output_nodes=3).to(device)
angles_model = WeightedGraphNet(dataset[0][1],
    iterations=run_parameters["iterations"], out_channels=run_parameters["out_channels"],
    hidden_output_nodes=6, output_nodes=3).to(device)
dihedrals_model = WeightedGraphNet(dataset[0][2],
    iterations=run_parameters["iterations"], out_channels=run_parameters["out_channels"],
    hidden_output_nodes=6, output_nodes=3).to(device)

model = MultiGraphNet([interactions_model, angles_model, dihedrals_model]).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
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
        train_samples = [dataset[j][i].to(device) for i in range(len(dataset[0]))]
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(train_samples)

        # Compute and print loss
        loss = criterion(y_pred, train_samples[0].y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if number % 500 == 0 and number > 0:
            print("{} samples out of {}".format(number, len(train_ind)))

    loss = torch.mean(torch.as_tensor(losses)).item()
    if NORMALIZE_DATA:
        loss = loss*target_std
    print("Epoch {} - Training loss: {:.2f}".format(i, loss))


predictions = []
errors = []
model.eval()
for j in test_ind:
    test_sample = [dataset[j][i].to(device) for i in range(len(dataset[0]))]
    # Forward pass: Compute predicted y by passing x to the model
    prediction = model(test_sample)
    error = prediction - test_sample[0].y
    predictions.append(prediction.item())
    errors.append(error.item())

# Compute MAE
mae = np.absolute(np.asarray(errors)).mean()
if NORMALIZE_DATA:
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
# !!!!. Visualize FES output of the NN (I have something very close to it)
# !!!!. Change representation of the error (consider diagonal instead of horizontal)
# !. Read paper on NNConv after having read slides from geometric deep learning
# !. Understand what is graph attention and try pooling methods (top-k?)
# GOOD: (invert the graph) good information in edges: find a way to prioritize them
# GOOD: Convert angles in the cosine form
# (....) Try HypergraphConv layer

# ? (To investigate) use dataset batching https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
# ?. Use validation set
# ?. Add dropout layer if needed
# ?. check if should use edge_weight or edge_attributes

# Plus: Understand what pos= on Data means (should not change since I don't use it in the FirstGraphNet)

