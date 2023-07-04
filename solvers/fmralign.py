from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    from benchmark_utils import gradient_ols
    from benchopt.stopping_criterion import SingleRunCriterion
    from fmralign.pairwise_alignment import PairwiseAlignment
    from joblib import Memory


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "fmralign"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'method': ['identity'],
    }

    stopping_criterion = SingleRunCriterion()

    def set_objective(self, source, target, mask):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.source = source
        self.target = target
        self.mask = mask

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.`

        print("Running the solver\n")

        alignement_estimator = PairwiseAlignment(
            alignment_method=self.method,
            n_pieces=1,
            mask=self.mask,
            memory=Memory(),
            memory_level=1,
        ).fit(self.source, self.target)
        self.alignment_estimator = alignement_estimator

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.

        return self.alignment_estimator
