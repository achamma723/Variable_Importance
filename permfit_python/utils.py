import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def create_X_y(
    X,
    y,
    bootstrap=True,
    split_perc=0.8,
    prob_type="regression",
    list_cont=[],
    random_state=None,
):
    """Create train/valid split of input data X and target variable y.
    Parameters
    ----------
    X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The input samples before the splitting process.
    y: ndarray, shape (n_samples, )
        The output samples before the splitting process.
    bootstrap: bool, default=True
        Application of bootstrap sampling for the training set.
    split_perc: float, default=0.8
        The training/validation cut for the provided data.
    prob_type: str, default='regression'
        A classification or a regression problem.
    list_cont: list, default=[]
        The list of continuous variables.
    random_state: int, default=2023
        Fixing the seeds of the random generator.

    Returns
    -------
    X_train_scaled: {array-like, sparse matrix}, shape (n_train_samples, n_features)
        The bootstrapped training input samples with scaled continuous variables.
    y_train_scaled: {array-like}, shape (n_train_samples, )
        The bootstrapped training output samples scaled if continous.
    X_valid_scaled: {array-like, sparse matrix}, shape (n_valid_samples, n_features)
        The validation input samples with scaled continuous variables.
    y_valid_scaled: {array-like}, shape (n_valid_samples, )
        The validation output samples scaled if continous.
    X_scaled: {array-like, sparse matrix}, shape (n_samples, n_features)
        The original input samples with scaled continuous variables.
    y_valid: {array-like}, shape (n_samples, )
        The original output samples with validation indices.
    scaler_x: scikit-learn StandardScaler
        The standard scaler encoder for the continuous variables of the input.
    scaler_y: scikit-learn StandardScaler
        The standard scaler encoder for the output if continuous.
    valid_ind: list
        The list of indices of the validation set.
    """
    rng = np.random.RandomState(random_state)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    n = X.shape[0]

    if bootstrap:
        train_ind = rng.choice(n, size=n, replace=True)
    else:
        train_ind = rng.choice(
            n, size=int(np.floor(split_perc * n)), replace=False
        )
    valid_ind = np.array([ind for ind in range(n) if ind not in train_ind])

    X_train, X_valid = X[train_ind], X[valid_ind]
    y_train, y_valid = y[train_ind], y[valid_ind]

    # Scaling X and y
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    X_scaled = X.copy()

    if len(list_cont) > 0:
        X_train_scaled[:, list_cont] = scaler_x.fit_transform(
            X_train[:, list_cont]
        )
        X_valid_scaled[:, list_cont] = scaler_x.transform(
            X_valid[:, list_cont]
        )
        X_scaled[:, list_cont] = scaler_x.transform(X[:, list_cont])
    if prob_type == "regression":
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_valid_scaled = scaler_y.transform(y_valid)
    else:
        y_train_scaled = y_train.copy()
        y_valid_scaled = y_valid.copy()

    return (
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        X_scaled,
        y_valid,
        scaler_x,
        scaler_y,
        valid_ind,
    )


def sigmoid(x):
    """The function applies the sigmoid function element-wise to the input array x."""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """The function applies the relu function element-wise to the input array x."""
    return (abs(x) + x) / 2


def relu_(x):
    """The function applies the derivative of the relu function element-wise
    to the input array x.
    """
    return (x > 0) * 1


class RandomForestClassifierModified(RandomForestClassifier):
    def fit(self, X, y):
        self.y_ = y
        super().fit(X, y)

    def predict(self, X):
        # Get the leaf indices for each sample in the input data
        leaf_indices = self.apply(X)

        # Initialize an array to store the predictions for each sample
        predictions = np.empty(
            (X.shape[0], self.n_estimators), dtype=self.classes_.dtype
        )

        # Loop over each sample in the input data
        for i in range(X.shape[0]):
            # Removing the row of the corresponding input sample
            leaf_indices_minus_i = np.delete(leaf_indices, i, axis=0)

            # The list of indices sampled from the same leaf of the input sample
            leaf_samples = []
            # Loop over each tree in the forest
            for j in range(self.n_estimators):
                # Find the samples that fall into the same leaf node for this tree
                samples_in_leaf = np.where(
                    leaf_indices_minus_i[:, j] == leaf_indices[i, j]
                )[0]

                # Append the samples to the list
                leaf_samples.append(self.y_[samples_in_leaf])

            predictions[i, :] = np.ravel(self.y_[np.array(leaf_samples)])

        # Combine the predictions from all trees to make the final prediction
        return np.apply_along_axis(
            lambda row: np.argmax(np.bincount(row)),
            axis=1,
            arr=predictions,
        )


def ordinal_encode(y):
    """This function encodes the ordinal variable with a special gradual encoding storing also
    the natural order information.
    """
    list_y = []
    for y_col in range(y.shape[-1]):
        # create a zero-filled array for the ordinal encoding
        y_ordinal = np.zeros((len(y[:, y_col]), len(set(y[:, y_col]))))
        # set the appropriate indices to 1 for each ordinal value and all lower ordinal values
        for ind_el, el in enumerate(y[:, y_col]):
            y_ordinal[ind_el, np.arange(el + 1)] = 1
        list_y.append(y_ordinal[:, 1:])

    return list_y


class RandomForestRegressorModified(RandomForestRegressor):
    def fit(self, X, y):
        self.y_ = y
        super().fit(X, y)

    def predict(self, X):
        rng = np.random.RandomState(2023)

        # Get the leaf indices for each sample in the input data
        leaf_indices = self.apply(X)

        # Initialize an array to store the predictions for each sample
        predictions = []

        # Loop over each sample in the input data
        for i in range(X.shape[0]):
            # Removing the row of the corresponding input sample
            leaf_indices_minus_i = np.delete(leaf_indices, i, axis=0)

            # The list of indices sampled from the same leaf of the input sample
            leaf_samples = []
            # Loop over each tree in the forest
            for j in range(self.n_estimators):
                # Find the samples that fall into the same leaf node for this tree
                samples_in_leaf = np.where(
                    leaf_indices_minus_i[:, j] == leaf_indices[i, j]
                )[0]

                # Append the samples to the list
                leaf_samples.append(self.y_[samples_in_leaf])

            predictions.append(leaf_samples)

        # Combine the predictions from all trees to make the final prediction
        return predictions


def sample_predictions(list_samples, random_state=None):
    rng = np.random.RandomState(random_state)
    predictions = np.zeros((len(list_samples), len(list_samples[0]), 1))

    for ind_input, input_sample in enumerate(list_samples):
        for ind_sample_estimator, sample_estimator in enumerate(
            input_sample
        ):
            predictions[ind_input, ind_sample_estimator] = rng.choice(
                sample_estimator
            )
    return predictions
