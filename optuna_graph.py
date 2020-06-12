# system imports
import pickle
import numpy as np
from warnings import warn
import random
import json
from datetime import datetime
import time
import os
from pprint import pprint
import tqdm

# pytorch imports
import torch
from torch.nn import L1Loss
import torch.optim as optim
from torch_geometric.data import Data
import optuna

# Custom imports
from helpers import mol2graph
from helpers.EarlyStopping import EarlyStopping
from helpers.scale import normalize
from GraphPoolingNets import TopKPoolingNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    warn("You are using CPU instead of CUDA. The computation will be longer...")


seed = 123454
# seed = 76583
random.seed(seed)
torch.manual_seed(seed)

# Data parameters
DATASET_TYPE = "old"
DATA_DIR = f"ala_dipep_{DATASET_TYPE}"
TARGET_FILE = f"free-energy-{DATASET_TYPE}.dat"
N_SAMPLES = 3815 if DATASET_TYPE == "small" else 21881 if DATASET_TYPE == "medium" else 50000 if DATASET_TYPE == "old" else 64074 if DATASET_TYPE == "big" else 48952
NORMALIZE_DATA = True
NORMALIZE_TARGET = True
OVERWRITE_PICKLES = False
UNSEEN_REGION = None  # can be "left", "right" or None. When is "left" we train on "right" and predict on "left"

if not OVERWRITE_PICKLES:
    warn("You are using EXISTING pickles, change this setting if you add features to nodes/edges ")

# Parameters
run_parameters = {
    "seed": seed,
    "sin_cos": True,
    "graph_type": "De Bruijn",
    "out_channels": 4,
    "convolution": "GraphConv",
    "convolutions": 3,
    "learning_rate": 0.0001 if NORMALIZE_TARGET else 0.001,
    "epochs": 2000,
    "patience": 30,
    "normalize_target": NORMALIZE_TARGET,
    "dataset_perc": 1,
    "shuffle": False,
    "train_split": 0.2,
    "validation_split": 0.1,
    "unseen_region": UNSEEN_REGION
}
stopping = EarlyStopping(patience=run_parameters["patience"])


def read_dataset(train_perc):
    run_parameters["train_split"] = train_perc
    if UNSEEN_REGION is not None:
        seen_region = 'right' if UNSEEN_REGION == 'left' else 'left'
        warn(f"Training on {seen_region} minima only. Testing on {UNSEEN_REGION} minima.")

        with open(f"{DATA_DIR}/left.json", "r") as l:
            left = json.load(l)

        with open(f"{DATA_DIR}/right.json", "r") as r:
            right = json.load(r)

        # Training on everything else
        if UNSEEN_REGION == "left":
            indexes = left
        else:
            indexes = right

        train_ind = [i for i in range(N_SAMPLES) if i not in indexes]
        # half to validation, half to test
        random.shuffle(indexes)
        split = np.int(0.5 * len(indexes))
        validation_ind = indexes[:split]
        test_ind = indexes[split:]
    else:
        indexes = [i for i in range(N_SAMPLES)]
        random.shuffle(indexes)
        indexes = indexes[:np.int(run_parameters["dataset_perc"] * N_SAMPLES)]
        split = np.int(run_parameters["train_split"] * len(indexes))
        train_ind = indexes[:split]
        split_2 = split + np.int(run_parameters["validation_split"] * len(indexes))
        validation_ind = indexes[split:split_2]
        test_ind = indexes[split_2:]

    graph_samples = []
    for i in range(N_SAMPLES):
        try:
            if OVERWRITE_PICKLES:
                raise FileNotFoundError

            with open("{}/{}-dihedrals-graph.pickle".format(DATA_DIR, i), "rb") as p:
                debruijn = pickle.load(p)

        except FileNotFoundError:
            atoms, edges, angles, dihedrals = mol2graph.get_richgraph("{}/{}.json".format(DATA_DIR, i))

            debruijn = mol2graph.get_central_overlap_graph(atoms, angles, dihedrals, shuffle=run_parameters["shuffle"],
                                                           sin_cos_decomposition=run_parameters["sin_cos"])

            if OVERWRITE_PICKLES:
                with open("{}/{}-dihedrals-graph.pickle".format(DATA_DIR, i), "wb") as p:
                    pickle.dump(debruijn, p)

        graph_samples.append(debruijn)

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

    return dataset, train_ind, validation_ind, test_ind, target_mean, target_std


def define_model(trial, sample):
    pooling_layers = trial.suggest_int("topk_pooling", 0, 2)
    topk_ratio = trial.suggest_discrete_uniform('topk_ratio', 0.4, 0.8, 0.1)
    final_pooling = trial.suggest_categorical("final_pooling", ["avg_pool_x", "sort_pooling"])
    dense_output = trial.suggest_categorical("dense_output", [True, False])
    channels_optuna = trial.suggest_int("channels_optuna", 1, 3)
    print({
        "channels_optuna": channels_optuna,
        "dense_output": dense_output,
        "final_pooling": final_pooling,
        "topk_ratio": topk_ratio,
        "pooling_layers": pooling_layers,
    })
    return TopKPoolingNet(sample, pooling_layers, topk_ratio, final_pooling, dense_output, channels_optuna)


def objective(trial):
    train_perc = trial.suggest_discrete_uniform("training_percentage", 0.05, 0.1, 0.05)
    dataset, train_ind, validation_ind, test_ind, target_mean, target_std = read_dataset(train_perc)
    model = define_model(trial, dataset[0]).to(device)
    criterion = L1Loss()
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    if optimizer_name == "SGD":
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, momentum=0.8)
    else:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    print({
        "lr": lr,
        "optimizer": optimizer,
    })
    start = time.time()
    for i in range(run_parameters["epochs"]):
        model.train()
        random.shuffle(train_ind)
        train_losses = []
        for number, j in enumerate(tqdm.tqdm(train_ind)):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(dataset[j].to(device))

            # Compute and print loss
            loss = criterion(y_pred, dataset[j].y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = torch.mean(torch.as_tensor(train_losses)).item()
        if NORMALIZE_TARGET:
            train_loss = train_loss * target_std
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
                val_loss = val_loss * target_std
            print("Epoch {} - Validation MAE: {:.2f} - Train MAE: {:.2f}".format(i + 1, val_loss, train_loss))

            # Check Early Stopping
            if stopping.check(val_loss):
                run_parameters["epochs"] = i + 1
                print(f"Training finished because of early stopping. Best loss on validation: {stopping.best_score}")
                break

        trial.report(val_loss, i+1)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    duration = (time.time() - start) / 60.0  # Minutes
    hours = np.int(np.floor(duration / 60.0))
    minutes = np.int(np.floor(duration - hours * 60))

    # No test, only validation
    #test(trial, model, dataset, test_ind)

    return val_loss


study_name = "TopK - avg_pool_x"
study = optuna.create_study(study_name=study_name, storage='sqlite:///dontknowhereitislocated.db', load_if_exists=True)
study.optimize(objective, n_trials=5)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
best = study.best_trial

print("  Value: ", best.value)

print("  Params: ")
for key, value in best.params.items():
    print("    {}: {}".format(key, value))


