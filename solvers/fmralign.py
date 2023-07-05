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
    from fastsrm.identifiable_srm import IdentifiableFastSRM
    from benchmark_utils.config import ROOT_FOLDER
    import os

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "fmralign"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'method': ['identity', 'fastsrm', 'scaled_orthogonal', 'ridge_cv']
    }

    stopping_criterion = SingleRunCriterion()

    def set_objective(self, source, source_subjects, target, mask):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.source = source
        self.source_subjects = source_subjects
        self.target = target
        self.mask = mask

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.`

        print("Running the solver\n")
        dict_alignment = {}
        if self.method == 'fastsrm':
            srm_path = os.path.join(ROOT_FOLDER, 'fastsrm')
            if not os.path.exists(srm_path):
                os.makedirs(srm_path)
            srm = IdentifiableFastSRM(n_components=50,
                                      aggregate="mean",
                                      temp_dir=srm_path, 
                                      tol=1e-10,
                                      n_iter=100,
                                      n_jobs=5)
            imgs_list = [self.mask.transform(contrasts).T for contrasts in self.source]
            print("Fitting SRM")
            alignment_estimator = srm.fit(
                imgs_list
            )
            for sub in self.source_subjects:
                dict_alignment[sub] = alignment_estimator
        else:
            for contrasts, sub in zip(self.source, self.source_subjects):
                alignment_estimator = PairwiseAlignment(
                    alignment_method=self.method,
                    n_pieces=150,
                    mask=self.mask,
                    memory=Memory(),
                    memory_level=1,
                ).fit(contrasts, self.target)
                dict_alignment[sub] = alignment_estimator
        self.alignment_estimators = dict_alignment

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        output_dict = dict(alignment_estimators=self.alignment_estimators,
                    method=self.method,
                    )

        return output_dict
