# system imports
import pickle
import numpy as np
from warnings import warn

# pytorch imports
import torch
from torch.nn import L1Loss, MSELoss
from torch_geometric.data import Data
from torch.utils.data import DataLoader

# Custom imports
import mol2graph
from scale import normalize
from GraphNet import WeightedGraphNet, MultiGraphNet

assert torch.__version__ == "1.5.0"  # Needed for pytorch-geometric


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    warn("You are using CPU instead of CUDA. The computation will be longer...")

seed = 76583
torch.manual_seed(seed)

# Data parameters
DATA_DIR = "ala_dipep_small"
TARGET_FILE = "free-energy-small.dat"
N_SAMPLES = 3815
# n_samples = 21881
NORMALIZE_DATA = True

# Hyperparameters
epochs = 10
criterion = L1Loss()
# criterion = MSELoss()

graph_samples = []
for i in range(N_SAMPLES):
    try:
        with open("{}/{}.pickle".format(DATA_DIR, i), "rb") as p:
            graph_sample = pickle.load(p)

    except FileNotFoundError:
        atoms, edges, angles, dihedrals = mol2graph.get_richgraph("{}/{}.json".format(DATA_DIR, i))

        interactions_sample = mol2graph.get_atoms_interactions_graph(atoms, edges)
        angles_sample = mol2graph.get_angles_graph(atoms, angles)
        dihedrals_sample = mol2graph.get_dihedrals_graph(atoms, dihedrals)

        graph_sample = [interactions_sample, angles_sample, dihedrals_sample]
        with open("{}/{}.pickle".format(DATA_DIR, i), "wb") as p:
            pickle.dump(graph_sample, p)

    graph_samples.append(graph_sample)

if NORMALIZE_DATA:
    columns = [[samples[i] for samples in graph_samples] for i in range(len(graph_samples[0]))]
    normalized_columns = [normalize(column) for column in columns]
    graph_samples = [[column[i] for column in normalized_columns] for i in range(len(graph_samples))]

with open(TARGET_FILE, "r") as t:
    target = [torch.tensor([float(v)]) for v in t.readlines()][:N_SAMPLES]

dataset = []
for i, samples in enumerate(graph_samples):
    dataset.append(
        [Data(x=sample[0], edge_index=sample[1], edge_attr=sample[2], y=target[i]).to(device) for sample in samples]
    )
print("Dataset loaded")


train_ind, validation_ind, test_ind = [], [], []

for i in range(0, len(dataset), 10):
    # if we are exceeding indexes, put these samples to training
    if i+10 > len(dataset):
        train_ind = train_ind + list(range(i, len(dataset)))
    else:
        train_ind = train_ind + list(range(i, i+8))
        validation_ind.append(i+8)
        test_ind.append(i+9)


interactions_model = WeightedGraphNet(dataset[0][0], iterations=2, output_nodes=2).to(device)
angles_model = WeightedGraphNet(dataset[0][1], iterations=2, output_nodes=2).to(device)
dihedrals_model = WeightedGraphNet(dataset[0][2], iterations=2, output_nodes=2).to(device)

model = MultiGraphNet([interactions_model, angles_model, dihedrals_model]).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                        factor=0.7, patience=5,
#                                                        min_lr=0.00001)
# TODO: print a good summary of the model
model.train()
print(model)
for i in range(epochs):
    # Investigate what this does
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

    print(i, torch.mean(torch.as_tensor(losses)))


predictions = []
errors = []
model.eval()
for j in test_ind:
    test_sample = [dataset[j][i].to(device) for i in range(len(dataset[0]))]
    # Forward pass: Compute predicted y by passing x to the model
    prediction = model(test_sample)
    predictions.append(prediction)
    error = prediction - test_sample[0].y
    errors.append(error.item())

# Compute MAE
print("Mean Absolute Error: {:.2f}".format(np.absolute(np.asarray(errors)).mean()))


# 1. (To investigate) use dataset batching https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
# 2. Use validation set
# 3. Print out with JSON format the results, print the config too
# 4. Add dropout layer if needed

# 5. check if should use edge_weight or edge_attributes

#############################
# 4. Read paper on NNConv!!!!
# Understand what is graph attention and try pooling methods (top-k?)
# Do HypergraphConv


# 5. Add correlation between predictions and target

# Then do the test on separate training

# Plus: Understand what pos= on Data means (should not change since I don't use it in the FirstGraphNet)

