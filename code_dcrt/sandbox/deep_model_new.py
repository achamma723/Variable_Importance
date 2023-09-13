import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torchmetrics import Accuracy
from torchsummary import summary
from tqdm import tqdm

accuracy = Accuracy()


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data = (layer.weight.data.uniform_() - 0.5) * 0.2
        layer.bias.data = (layer.bias.data.uniform_() - 0.5) * 0.1


def access_weights(model):
    l1_parameters = []
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            l1_parameters.append(layer.weight.view(-1))
    return torch.cat(l1_parameters)


class DNN(nn.Module):
    """Feedfoward neural network with 4 hidden layers"""

    def __init__(self, in_size):
        super().__init__()
        self.layers = nn.Sequential(
            # hidden layers
            nn.Linear(in_size, 50),
            nn.ELU(),
            nn.Linear(50, 40),
            nn.ELU(),
            nn.Linear(40, 30),
            nn.ELU(),
            nn.Linear(30, 20),
            nn.ELU(),
            # output layer
            nn.Linear(20, 1),
        )
        self.loss = 0
        self.pred = []
        self.gradients = []

    def forward(self, x):
        return self.layers(x)

    def compute_l1_loss(self, l1_weight):
        weights = access_weights(self)
        return l1_weight * torch.abs(weights).sum()

    def compute_l2_loss(self, l2_weight):
        weights = access_weights(self)
        return l2_weight * torch.pow(weights, 2).sum()

    def training_step(self, batch, device, prob_type):
        X, y = batch[0].to(device), batch[1].to(device)
        y_pred = self(X)  # Generate predictions
        if prob_type == "regression":
            loss = F.mse_loss(y_pred, y)
        else:
            loss = F.binary_cross_entropy_with_logits(y_pred, y)  # Calculate loss
        return loss

    def validation_step(self, batch, device, prob_type):
        X, y = batch[0].to(device), batch[1].to(device)
        y_pred = self(X)  # Generate predictions
        if prob_type == "regression":
            loss = F.mse_loss(y_pred, y)
            return {"val_mse": loss, "pred_val": y_pred, "batch_size": len(X), "X": X}
        else:
            loss = F.binary_cross_entropy_with_logits(y_pred, y)  # Calculate loss
            acc = accuracy(y_pred, y.int())
            return {
                "val_loss": loss,
                "val_acc": acc,
                "pred_val": y_pred,
                "batch_size": len(X),
                "X": X,
            }

    def validation_epoch_end(self, outputs, prob_type):
        self.pred = torch.cat([x["pred_val"] for x in outputs], dim=0)
        if prob_type == "classification":
            batch_losses = []
            batch_accs = []
            batch_sizes = []
            for x in outputs:
                batch_losses.append(x["val_loss"] * x["batch_size"])
                batch_accs.append(x["val_acc"] * x["batch_size"])
                batch_sizes.append(x["batch_size"])
            self.loss = torch.stack(batch_losses).sum().item() / np.sum(
                batch_sizes
            )  # Combine losses
            epoch_acc = torch.stack(batch_accs).sum().item() / np.sum(
                batch_sizes
            )  # Combine accuracies
            return {"val_loss": self.loss, "val_acc": epoch_acc}
        else:
            batch_losses = [x["val_mse"] * x["batch_size"] for x in outputs]
            batch_sizes = [x["batch_size"] for x in outputs]
            self.loss = torch.stack(batch_losses).sum().item() / np.sum(
                batch_sizes
            )  # Combine losses
            return {"val_mse": self.loss}

    def epoch_end(self, epoch, result):
        if len(result) == 2:
            print(
                "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch + 1, result["val_loss"], result["val_acc"]
                )
            )
        else:
            print("Epoch [{}], val_mse: {:.4f}".format(epoch + 1, result["val_mse"]))


def evaluate(model, loader, device, prob_type):
    outputs = [model.validation_step(batch, device, prob_type) for batch in loader]
    return model.validation_epoch_end(outputs, prob_type)


def evaluate_final(model, loader, device, prob_type):
    if isinstance(device, str):
        device = torch.device(device)
    outputs = [model.validation_step(batch, device, prob_type) for batch in loader]
    model.gradients = (
        torch.cat(
            [
                torch.autograd.grad(
                    x["pred_val"],
                    x["X"],
                    torch.ones_like(x["pred_val"]),
                    create_graph=True,
                )[0]
                for x in outputs
            ],
            dim=0,
        )
        .cpu()
        .detach()
        .numpy()
    )
    model.pred = torch.cat([x["pred_val"] for x in outputs], dim=0)
    model.pred = torch.squeeze(model.pred).cpu().detach().numpy()


def train_dnn(
    train_loader,
    val_loader,
    original_loader,
    p=50,
    n_epochs=200,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    l1_weight=1e-4,
    l2_weight=0,
    save_file=None,
    verbose=2,
    random_state=2021,
    prob_type="regression",
):
    """
    train_loader: DataLoader for Train data
    val_loader: DataLoader for Validation data
    original_loader: DataLoader for Original_data
    p: Number of variables
    n_epochs: The number of epochs
    lr: learning rate
    beta1: Beta1 parameter for Adam optimizer
    beta2: Beta2 parameter for Adam optimizer
    eps: Epsilon parameter for Adam optimizer
    l1_weight: L1 regalurization weight
    l2_weight: L2 regularization weight
    save_file: Save file for Best pytorch model
    verbose: If > 2, the metrics will be printed
    prob_type: A classification or regression problem
    """
    # Set fixed random number seed
    # torch.manual_seed(random_state)
    # Specify whether to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy.to(device)
    # DNN model
    model = DNN(p)
    model.to(device)
    # Initializing weights/bias
    model.apply(init_weights)
    # Adam Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps
    )

    best_loss = 1e100
    best_model = DNN(p)
    best_epoch = 1
    for epoch in range(n_epochs):
        # Training Phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = (
                model.training_step(batch, device, prob_type)
                + model.compute_l1_loss(l1_weight)
                + model.compute_l2_loss(l2_weight)
            )
            loss.backward()
            optimizer.step()
        # Validation Phase
        model.eval()
        result = evaluate(model, val_loader, device, prob_type)
        if model.loss < best_loss:
            best_loss = model.loss
            best_model = torch.save(model, save_file + ".pth")
            best_epoch = epoch + 1
        if verbose >= 2:
            model.epoch_end(epoch, result)

    best_model = torch.load(save_file + ".pth")
    evaluate_final(best_model, original_loader, device, prob_type)
    return best_model


def _ensemble_dnnet(
    X,
    y,
    n_ensemble=100,
    verbose=1,
    bootstrap=True,
    split_perc=0.8,
    batch_size=32,
    batch_size_val=128,
    min_keep=10,
    prob_type="regression",
    save_file=None,
    random_state=2021,
    use_cnn=False,
):
    """
    X: The matrix of predictors
    y: The response vector
    n_ensemble: The number of DNNs to be fit
    verbose: If > 1, the progress bar will be printed
    bootstrap: If True, a bootstrap sampling is used
    split_perc: If bootstrap==False, a training/validation cut for the data will be used
    batch_size: The number of samples per batch for training
    batch_size_val: The number of samples per batch for validation
    min_keep: The minimal number of DNNs to be kept
    prob_type: A classification or regression problem
    save_file: Save file for Best pytorch model
    random_state: Fixing the seeds of the random generator
    """
    n, p = X.shape
    min_keep = max(min(min_keep, n_ensemble), 1)

    if verbose >= 1:
        pbar = tqdm(total=n_ensemble)

    scalers_list = [(StandardScaler(), StandardScaler()) for s in range(n_ensemble)]
    models_list = []
    loss = np.empty(n_ensemble)
    pred = np.empty((n_ensemble, n))
    list_gradients = []
    for i in range(n_ensemble):
        # Sampling and Train/Validate splitting
        if bootstrap:
            train_ind = np.random.choice(n, size=n, replace=True)
        else:
            train_ind = np.random.choice(
                n, size=int(np.floor(split_perc * n)), replace=False
            )
        valid_ind = np.array([ind for ind in range(n) if ind not in train_ind])

        X_train, X_valid = X[train_ind], X[valid_ind]
        y_train, y_valid = y[train_ind], y[valid_ind]

        # Scaling X and y
        X_train_scaled = scalers_list[i][0].fit_transform(X_train)
        X_valid_scaled = scalers_list[i][0].transform(X_valid)
        X_scaled = scalers_list[i][0].transform(X)

        if prob_type == "regression":
            y_train_scaled = scalers_list[i][1].fit_transform(y_train)
            y_valid_scaled = scalers_list[i][1].transform(y_valid)
            y_scaled = scalers_list[i][1].transform(y)
        else:
            y_train_scaled = y_train.copy()
            y_valid_scaled = y_valid.copy()
            y_scaled = y.copy()

        tensor_y_train = torch.from_numpy(y_train_scaled).float()
        tensor_y_valid = torch.from_numpy(y_valid_scaled).float()
        tensor_y = torch.from_numpy(y_scaled).float()

        # Creating DataLoaders
        train_loader = Dataset_Loader(
            X_train_scaled, tensor_y_train, shuffle=True, batch_size=batch_size
        )
        validate_loader = Dataset_Loader(
            X_valid_scaled, tensor_y_valid, batch_size=batch_size_val
        )
        original_loader = Dataset_Loader(X_scaled, tensor_y, batch_size=batch_size_val)
        current_model = train_dnn(
            train_loader,
            validate_loader,
            original_loader,
            p=X_train.shape[1],
            save_file=save_file,
            verbose=verbose,
            random_state=random_state,
            prob_type=prob_type,
        )
        models_list.append(current_model)
        list_gradients.append(current_model.gradients)

        res = np.mean(np.concatenate(list_gradients), axis=0)
        print(res)
        exit(0)

        if prob_type == "regression":
            pred[i, :] = (
                current_model.pred * scalers_list[i][1].scale_
                + scalers_list[i][1].mean_
            )
            loss[i] = np.std(y_valid) ** 2 - mean_squared_error(
                y_valid, pred[i, valid_ind]
            )
        else:
            pred[i, :] = sigmoid(current_model.pred)
            loss[i] = log_loss(
                y_valid, np.ones(len(y_valid)) * np.mean(y_valid)
            ) - log_loss(y_valid, pred[i, valid_ind])
        if verbose >= 1:
            pbar.update(1)

    if verbose >= 1:
        pbar.close()
    if n_ensemble == 1:
        return [(models_list[0], scalers_list[0])]
    # Keeping the optimal subset of DNNs
    sorted_loss = loss.copy()
    sorted_loss.sort()
    new_loss = np.empty(n_ensemble - 1)
    for i in range(n_ensemble - 1):
        current_pred = np.mean(pred[loss >= sorted_loss[i], :], axis=0)
        if prob_type == "regression":
            new_loss[i] = mean_squared_error(y, current_pred)
        else:
            new_loss[i] = log_loss(y, current_pred)
    keep_dnn = loss >= sorted_loss[np.argmin(new_loss[: (n_ensemble - min_keep + 1)])]
    return [
        (models_list[i], scalers_list[i])
        for i in range(n_ensemble)
        if keep_dnn[i] == True
    ]


def Dataset_Loader(X, tensor_y, shuffle=False, batch_size=50):
    tensor_x = torch.from_numpy(X).float()
    tensor_x.requires_grad = True
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
