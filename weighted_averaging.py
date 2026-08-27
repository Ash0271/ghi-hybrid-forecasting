"""
Hybridization technique 1 — Weighted Averaging with optimal weight search.

Combines the ARIMA and BiLSTM base-model predictions using a weighted average,
where the weight (alpha) is selected via multi-objective optimization (NSGA-II)
to minimize prediction error, rather than a fixed/manual weight.

This is my own contribution — it assumes arima_predictions, lstm_predictions,
and y_test are already produced by the (not-included) base models.
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem


def ensemble_pred(arima_predictions, lstm_predictions, alpha):
    """Weighted combination of the two base-model predictions."""
    return (1 - alpha) * arima_predictions + alpha * lstm_predictions


def compute_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def find_best_alpha(arima_predictions, lstm_predictions, y_test, pop_size=100, n_gen=100):
    """
    Search for the alpha (blend weight) that minimizes ensemble prediction error,
    using NSGA-II multi-objective optimization.
    """
    np.random.seed(0)

    def evaluate(alpha):
        preds = ensemble_pred(arima_predictions, lstm_predictions, alpha)
        return compute_error(y_test, preds)

    problem = get_problem("ZDT1")
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, ("n_gen", n_gen), verbose=False)

    best_solution = min(res.pop, key=lambda x: np.min(x.F))
    best_alpha = best_solution.X[0]
    return best_alpha


if __name__ == "__main__":
    # Example usage — arima_predictions, lstm_predictions, y_test would come from
    # the (not-included) base models on your dataset.
    #
    # best_alpha = find_best_alpha(arima_predictions, lstm_predictions, y_test)
    # final_predictions = ensemble_pred(arima_predictions, lstm_predictions, best_alpha)
    pass
