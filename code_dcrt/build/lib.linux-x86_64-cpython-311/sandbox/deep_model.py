import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torchmetrics import Accuracy
from sklearn.metrics import mean_squared_error, log_loss, r2_score, roc_auc_score
accuracy = Accuracy(task='binary')


def _init_weights(layer):
    """ Helper function to _train_dnn

    This function helps to initialize the weights and the bias of each
    layer of the module.

    Args:
        layer (torch.nn): Layer of the DNN module (Conv1d, Linear,
        ...)
    """
    if isinstance(layer, nn.Linear):
        layer.weight.data = (layer.weight.data.uniform_() - 0.5) * 0.2
        layer.bias.data = (layer.bias.data.uniform_() - 0.5) * 0.1


def _access_weights(model):
    """ Helper function to _compute_l1_loss and _compute_l2_loss of
    class DNN.

    Args:
        model (class DNN): Instance of the DNN module

    Returns:
        (tensor): The weights of the instance in one tensor
    """
    l1_parameters = []
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            l1_parameters.append(layer.weight.view(-1))

    return torch.cat(l1_parameters)


class DNN(nn.Module):
    """ Class for Deep Neural Network module.

    This class aims to create instances of the DNN module with the
    pre-defined architecture.

    Args:
        nn ([type]): [description]

    Returns:
        (__main__.DNN): An instance of the DNN module
    """

    def __init__(self, in_size, nb_knockoffs):
        super().__init__()
        self.layers = nn.Sequential(
            # The Flatten layer will sequeeze the dimension 1
            nn.Flatten(start_dim=1),
            nn.Linear(nb_knockoffs*in_size, in_size),
            nn.ELU(),
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
            nn.Linear(20, 1))

        self.loss = 0
        self.gradients = []
        self.pred = []

    def forward(self, x):
        return self.layers(x)

    def _compute_l1_loss(self, l1_weight):
        """ Helper function to _train_dnn

        This function helps in the computation of the L1-loss using
        the weights of the module.

        Args:
            l1_weight (float): L1-regularization parameter

        Returns:
            (float): The computed L1-loss
        """
        weights = _access_weights(self)
        return l1_weight * torch.abs(weights).sum()

    def _compute_l2_loss(self, l2_weight):
        """ Helper function to _train_dnn

        This function helps in the computation of the L2-loss using
        the weights of the module.

        Args:
            l2_weight (float): L2-regularization parameter

        Returns:
            (float): The computed L2-loss
        """
        weights = _access_weights(self)
        return l2_weight * torch.pow(weights, 2).sum()

    def _training_step(self, batch, device, prob_type):
        """ Helper function to _train_dnn

        This function helps to perform one step of training per batch
        of data for the instance of the DNN module.

        Args:
            batch (list): A list of two tensors for the predictors and
            the outcome
            device (torch.device): The computation will be held on the
            "cpu" or the "gpu"
            prob_type (str, optional): A classification or regression
            problem.

        Returns:
            loss (float): The computed loss for the classification or
            the regression problem.
        """
        X, y = batch[0].to(device), batch[1].to(device)
        y_pred = self(X)  # Generate predictions
        if prob_type == 'regression':
            loss = F.mse_loss(y_pred, y)
        else:
            loss = F.binary_cross_entropy_with_logits(
                y_pred, y)  # Calculate loss
        return loss

    def _validation_step(self, batch, device, prob_type):
        """ Helper function to _train_dnn

        This function helps to perform one step of validation per
        batch of data for the instance of the DNN module.

        Args:
            batch (list): A list of two tensors for the predictors and
            the outcome
            device (torch.device): The computation will be held on the
            "cpu" or the "gpu"
            prob_type (str, optional): A classification or regression
            problem.

        Returns:
            (dict): The dictionary will return the loss, the
            predictions, the number of samples and the predictors of
            the current batch. If classification case, the accuracy
            also will be returned.
        """
        X, y = batch[0].to(device), batch[1].to(device)
        y_pred = self(X)  # Generate predictions
        if prob_type == 'regression':
            loss = F.mse_loss(y_pred, y)
            return {'val_mse': loss, 'pred_val': y_pred, 'batch_size': len(X), 'X': X}
        else:
            loss = F.binary_cross_entropy_with_logits(
                y_pred, y)  # Calculate loss
            acc = accuracy(y_pred, y.int())
            return {'val_loss': loss, 'val_acc': acc, 'pred_val': y_pred, 'batch_size': len(X), 'X': X}

    def _validation_epoch_end(self, outputs, prob_type):
        """ Helper function to _train_dnn

        This function helps to perform the concatenation of the losses
        for the batches of the validation set for the instance of the
        DNN module.

        Args:
            outputs (dict): [description]
            prob_type (str, optional): A classification or regression
            problem.

        Returns:
            (dict): The dictionary will return the loss for the whole
            validation set. If classification, the accuracy for the
            whole data will also be returned.
        """
        if prob_type == 'classification':
            batch_losses = []
            batch_accs = []
            batch_sizes = []
            for x in outputs:
                batch_losses.append(x['val_loss'] * x['batch_size'])
                batch_accs.append(x['val_acc'] * x['batch_size'])
                batch_sizes.append(x['batch_size'])
            self.loss = torch.stack(batch_losses).sum(
            ).item() / np.sum(batch_sizes)  # Combine losses
            epoch_acc = torch.stack(batch_accs).sum().item(
            ) / np.sum(batch_sizes)  # Combine accuracies
            return {'val_loss': self.loss, 'val_acc': epoch_acc}
        else:
            batch_losses = [x['val_mse'] * x['batch_size'] for x in outputs]
            batch_sizes = [x['batch_size'] for x in outputs]
            self.loss = torch.stack(batch_losses).sum(
            ).item() / np.sum(batch_sizes)  # Combine losses
            return {'val_mse': self.loss}

    def _epoch_end(self, epoch, result):
        """ Helper function to _train_dnn

        This function helps to print the progress at the end of each epoch.

        Args:
            epoch (int): The current epoch to print the progress for
            result (dict): A dictionary returning the loss at the end
            of the current epoch. If classification, the accuracy also
            will be returned
        """
        if len(result) == 2:
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch+1, result['val_loss'], result['val_acc']))
        else:
            print("Epoch [{}], val_mse: {:.4f}".format(
                epoch+1, result['val_mse']))


def _evaluate(model, loader, device, prob_type):
    """ Helper function to _train_dnn

    Args:
        model (__main__.DNN): An instance of the DNN module
        loader (Pytorch DataLoader): The loader containing the
        predictors and the outcome of the validation set
        device (torch.device): The computation will be held on the
        "cpu" or the "gpu"
        prob_type (str, optional): A classification or regression
        problem.

    Returns:
        (dict): The dictionary will return the loss for the whole
        validation set. If classification, the accuracy for the whole
        data will also be returned.
    """
    outputs = [model._validation_step(
        batch, device, prob_type) for batch in loader]
    return model._validation_epoch_end(outputs, prob_type)


def _evaluate_final(model, loader, device, prob_type):
    """ Helper function to _train_dnn
    Args:
        model (__main__.DNN): An instance of the DNN module
        loader (Pytorch DataLoader): The loader containing the
        predictors and the outcome of the whole dataset
        device (torch.device): The computation will be held on the
        "cpu" or the "gpu"
        prob_type (str, optional): A classification or regression
        problem.
    """
    if isinstance(device, str):
        device = torch.device(device)
    outputs = [model._validation_step(
        batch, device, prob_type) for batch in loader]
    model.pred = torch.cat([x['pred_val'] for x in outputs], dim=0).cpu().detach().numpy()
    model.gradients = torch.cat([torch.autograd.grad(x['pred_val'], x['X'], torch.ones_like(
        x['pred_val']), create_graph=True)[0] for x in outputs], dim=0).cpu().detach().numpy()
    # p_f = outputs[0]['X'].shape[2]
    # model.gradients = torch.eye(p_f * 2)
    # for layer in model.layers:
    #     if isinstance(layer, nn.Linear):
    #         model.gradients = torch.mm(model.gradients, torch.transpose(layer.weight, 0, 1))
    # model.gradients = model.gradients.cpu().detach().numpy().reshape(2, p_f)

def _train_dnn(train_loader, val_loader, test_loader, p=50, nb_knockoffs=5, n_epochs=200, lr=1e-3, beta1=0.9,
               beta2=0.999, eps=1e-8, l1_weight=1e-4, l2_weight=0, save_file=None, verbose=2,
               random_state=2021, prob_type='regression'):
    """ Helper function for _ensemble_dnnet

    This function helps to train one instance of the DNN module.
    The training process is realized with the training set using the
    "train_loader" according to the pre-defined architecture. The
    choice of the best model to keep (best epoch) is performed
    according to the comparison with the best loss attained so far on
    the validation set using the "val_loader". Finally, the prediction
    phase of the whole data is done using the "original_loader" and
    the predictions are saved into the class variable "pred". 

    Args:
        train_loader (Pytorch DataLoader): DataLoader for Train data
        val_loader (Pytorch DataLoader): DataLoader for Validation data
        original_loader (Pytorch DataLoader): DataLoader for Original_data
        p (int, optional): Number of variables. Defaults to 50.
        n_epochs (int, optional): The number of epochs. Defaults to
        250.
        lr (float, optional): learning rate. Defaults to 1e-3.
        beta1 (float, optional): Beta1 parameter for Adam optimizer.
        Defaults to 0.9.
        beta2 (float, optional): Beta2 parameter for Adam optimizer.
        Defaults to 0.999.
        eps (float, optional): Epsilon parameter for Adam optimizer.
        Defaults to 1e-8.
        l1_weight (float, optional): L1 regalurization weight.
        Defaults to 1e-4.
        l2_weight (int, optional): L2 regularization weight. Defaults
        to 0.
        save_file (str, optional): Filename to save the Best pytorch
        model. Defaults to None.
        verbose (int, optional): If > 2, the metrics will be printed.
        Defaults to 2.
        random_state (int, optional): Fixing the seed of the random
        generator. Defaults to 2021.
        prob_type (str, optional): A classification or regression
        problem. Defaults to 'regression'.

    Returns:
        (__main__.DNN): The best model attained after checking all the epochs.
    """
    # Set fixed random number seed
    # torch.manual_seed(random_state)
    # Specify whether to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy.to(device)
    # DNN model
    model = DNN(p, nb_knockoffs)
    model.to(device)
    # Initializing weights/bias
    model.apply(_init_weights)
    # Adam Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

    best_loss = 1e100
    best_model = DNN(p, nb_knockoffs)
    best_epoch = 1
    for epoch in range(n_epochs):
        # Training Phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model._training_step(batch, device, prob_type) + model._compute_l1_loss(
                l1_weight) + model._compute_l2_loss(l2_weight)
            loss.backward()
            optimizer.step()
        # Validation Phase
        model.eval()
        result = _evaluate(model, val_loader, device, prob_type)
        if model.loss < best_loss:
            best_loss = model.loss
            best_model = torch.save(model, save_file + '.pth')
            best_epoch = epoch+1
        if verbose >= 2:
            model._epoch_end(epoch, result)

    best_model = torch.load(save_file + '.pth')
    _evaluate_final(best_model, test_loader, device, prob_type)
    return best_model


def _ensemble_dnnet(X, y, X_test, y_test, n_ensemble=100, verbose=1, bootstrap=True, split_perc=0.8,
                    batch_size=32, batch_size_val=128, min_keep=10, prob_type='regression',
                    save_file=None, random_state=2021):
    """ Helper function to permfit

    This function helps to train the ensemble of DNN module's
    instances. Following, a filtration step is performed in order to
    keep the best between the best instances already trained. 

    Args:
        X (numpy.array):  The matrix of predictors
        y (numpy.array): The response vector
        n_ensemble (int, optional): The number of DNNs to be fit.
        Defaults to 100.
        verbose (int, optional): If > 1, the progress bar will be
        printed. Defaults to 1.
        bootstrap (bool, optional): If True, a bootstrap sampling is
        used. Defaults to True.
        split_perc (float, optional): If bootstrap==False, a
        training/validation cut for the data will be used. Defaults to
        0.8.
        batch_size (int, optional): The number of samples per batch
        for training. Defaults to 32.
        batch_size_val (int, optional): The number of samples per
        batch for validation. Defaults to 128.
        min_keep (int, optional): The minimal number of DNNs to be
        kept. Defaults to 10.
        prob_type (str, optional): A classification or regression
        problem. Defaults to 'regression'.
        save_file (str, optional): Filename to save the Best pytorch
        model. Defaults to None.
        random_state (int, optional): Fixing the seed of the random
        generator. Defaults to 2021.

    Returns:
        (list of tuples): The first element of each tuple is one of
        the chosen best models and the second element is the scalers
        list corresponding to this model.
    """
    n, m, p = X.shape
    scalers_list = [StandardScaler() for s in range(m)]
    knockoffs_pairs = np.array([np.array([i, j]) for i, j in zip(
        np.zeros(m-1, dtype=int), np.arange(1, m))])

    if bootstrap:
        train_ind = np.random.choice(n, size=n, replace=True)
    else:
        train_ind = np.random.choice(n, size=int(
            np.floor(split_perc*n)), replace=False)
    valid_ind = np.array([ind for ind in range(n) if ind not in train_ind])
    # Sampling and Train/Validate splitting
    X_train, X_valid = X[train_ind], X[valid_ind]
    y_train, y_valid = y[train_ind], y[valid_ind]

    X_train_scaled = np.zeros((X_train.shape[0], m, p))
    X_valid_scaled = np.zeros((X_valid.shape[0], m, p))
    X_test_scaled = np.zeros((X_test.shape[0], m, p))
    # Scaling the original features and the knockoffs
    for lay in range(m):
        X_train_scaled[:, lay, :] = scalers_list[lay].fit_transform(
            X_train[:, lay, :])
        X_valid_scaled[:, lay, :] = scalers_list[lay].transform(
            X_valid[:, lay, :])
        X_test_scaled[:, lay, :] = scalers_list[lay].transform(
            X_test[:, lay, :])
    # Scaling y
    if prob_type == 'regression':
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_valid_scaled = scaler_y.transform(y_valid)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        y_train_scaled = y_train.copy()
        y_valid_scaled = y_valid.copy()
        y_test_scaled = y_test.copy()

    tensor_y_train = torch.from_numpy(y_train_scaled).float()
    tensor_y_valid = torch.from_numpy(y_valid_scaled).float()
    tensor_y_test = torch.from_numpy(y_test_scaled).float()
    if verbose >= 1:
        pbar = tqdm(total=n_ensemble)

    list_gradients = []
    preds = np.empty((len(y_test), n_ensemble))

    for i in range(n_ensemble):
        # Creating DataLoaders
        train_loader = _Dataset_Loader(
            X_train_scaled[:, knockoffs_pairs[0], :], tensor_y_train, shuffle=True, batch_size=batch_size)
        validate_loader = _Dataset_Loader(
            X_valid_scaled[:, knockoffs_pairs[0], :], tensor_y_valid, batch_size=batch_size_val)
        test_loader = _Dataset_Loader(
            X_test_scaled[:, knockoffs_pairs[0], :], tensor_y_test, batch_size=batch_size_val)
        current_model = _train_dnn(train_loader, validate_loader, test_loader,
                                   p=X_train.shape[2], nb_knockoffs=2, save_file=save_file,
                                   verbose=verbose, random_state=random_state, prob_type=prob_type)
        
        list_gradients.append(current_model.gradients)

        if prob_type == 'regression':
            preds[:, i] = current_model.pred[:, 0] * \
                scaler_y.scale_ + scaler_y.mean_
        else:
            preds[:, i] = current_model.pred[:, 0]

        if verbose >= 1:
            pbar.update(1)

    if verbose >= 1:
        pbar.close()

    # Gradients case
    res = np.mean(np.concatenate(list_gradients), axis=0)
    # Weight case
    # res = np.mean(np.array(list_gradients), axis=0)

    if prob_type == "regression":
        score = r2_score(y_test, np.mean(preds, axis=1))
    else:
        score = roc_auc_score(y_test, _sigmoid(np.mean(preds, axis=1)))

    res = abs(res[0, :]) - abs(res[1, :])
    
    return res, score


def _Dataset_Loader(X, tensor_y, shuffle=False, batch_size=50):
    """ Helper function to _ensemble_dnnet

    This function helps to prepare the dataset loader with both the
    predictors and the outcome.

    Args:
        X (numpy.array): The array of predictors/variables
        tensor_y (tensor): The tensor of values of the outcome
        shuffle (bool, optional): If True, the data will be shuffled
        inside the loader. Defaults to False.
        batch_size (int, optional): The number of samples per batch. Defaults to 50.

    Returns:
        loader (torch.utils.data.DataLoader): A Pytorch Dataloader to use in the training/validation processes.
    """
    tensor_x = torch.from_numpy(X).float()
    tensor_x.requires_grad = True
    dataset = torch.utils.data.TensorDataset(tensor_x,
                                             tensor_y)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def _sigmoid(x):
    """ Helper function to _ensemble_dnnet

    Args:
        x (float or numpy.array of floats): A value or array of values
        to compute the sigmoid (inverse logit) for

    Returns:
        (float): A value in the interval [0, 1]
    """
    return 1 / (1 + np.exp(-x))
