import os
import typing
import sys

from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.kernel_approximation import Nystroem
import matplotlib.pyplot as plt
from matplotlib import cm


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        self.c = 0.75  # Hyperparameter for cost function optimization: Constant to scale "jumps" for prediction adjustment

        # TODO: Add custom initialization for your model here if necessary
        # self.kernel = DotProduct() + WhiteKernel()
        self.kernel = Matern(nu=0.5)
        # Init gaussian process regressor using defined kernel and normalize y
        self.gpm = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=4,
            normalize_y=True,
            random_state=self.rng.integers(low=0, high=10, size=1)[0],
        )

    def predict(
        self, x: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # x_test_transformed = self.feature_map_nystroem.fit_transform(x)
        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean, gp_std = self.gpm.predict(x, return_std=True)

        # TODO: Use the GP posterior to form your predictions here
        predictions = gp_mean.copy()
        # If gp_mean - standard deviation is above the THRESHOLD, we reduce the prediction
        # to avoid the overprediction error costs
        np.where(
            gp_mean - self.c * gp_std > THRESHOLD, gp_mean - self.c * gp_std, gp_mean
        )
        # If gp_mean is slightly under the threshold (1 std below), we replace it with the threshold to avoid
        # the high costs for underpredicting
        predictions[
            (THRESHOLD - gp_mean < self.c * gp_std) & (THRESHOLD - gp_mean > 0)
        ] = THRESHOLD

        return predictions, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # x_transformed = self.feature_map_nystroem.fit_transform(train_x)
        self.gpm.fit(train_x, train_y)

    def optimize_Hyperparameter(self, test_x: np.ndarray, test_y: np.ndarray) -> float:
        c_optim = self.c  # Init hyperparameter c
        min_costs = sys.float_info.max  # Init min_costs with max float value
        crt_costs = 0  # Init current costs with 0
        c_params = np.linspace(
            0.0, 2.0, num=22, endpoint=True
        )  # Init a parameter space in which we search for the optimal c
        # Iterate over all selected values for c to find value that minimizes cost function with "validation" set
        for c_param in c_params:
            self.c = c_param  # Set c to current value of c in iteration
            crt_predictions = self.predict(test_x)[
                0
            ]  # predict using the current c-value and only return predictions
            crt_costs = cost_function(
                test_y, crt_predictions
            )  # calculate costs using current c-value
            print(
                "costs for c=", c_param, "are", crt_costs
            )  # Print results for easy review of convergence/behaviour
            if crt_costs < min_costs:  # If costs are reduced, change optimal c-value
                c_optim = c_param  # change optimal c-value
                min_costs = crt_costs  # update min_costs
        print(
            "costs for c=", c_optim, "are", min_costs
        )  # Print best parameter and related costs
        self.c = c_optim  # Set c-parameter to optimal value


def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert (
        y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape
    )

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = "/results"):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print("Performing extended evaluation")
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle("Extended visualization of task 1")

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS)
        / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS)
        / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(
        predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    )
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title("Predictions")
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection="3d")
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False,
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title("GP means, colors are GP stddev")

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title("GP estimated stddev")
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, "extended_evaluation.pdf")
    fig.savefig(figure_path)
    print(f"Saved extended evaluation to {figure_path}")

    plt.show()


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt("train_x.csv", delimiter=",", skiprows=1)
    train_y = np.loadtxt("train_y.csv", delimiter=",", skiprows=1)
    test_x = np.loadtxt("test_x.csv", delimiter=",", skiprows=1)

    # Fit the model
    print("Fitting model")
    model = Model()

    # Select a random subset of the data to train model on and have a separate testset to compute score and optimize
    # hyperparameters (weight for prediction optimization)
    random_subset = model.rng.integers(
        low=0, high=len(train_y), size=int(0.7 * len(train_y))
    )

    model.fit_model(train_x[random_subset][:], train_y[random_subset])

    # Optimize Hyperparameters
    model.optimize_Hyperparameter(
        np.delete(train_x, random_subset, axis=0),
        np.delete(train_y, random_subset, axis=0),
    )

    # Predict on the test features
    print("Predicting on test features")
    predicted_y = model.predict(test_x)
    print(predicted_y)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir=".")


if __name__ == "__main__":
    main()
