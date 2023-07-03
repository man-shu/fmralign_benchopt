from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from fmralign.metrics import score_voxelwise


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "frmalign-benchopt"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'target_subject': ['sub-04'],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3"

    def set_data(self, alignment_dataset, projected_dataset, mask):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.alignment_dataset = alignment_dataset
        self.projected_dataset = projected_dataset
        self.mask = mask

        # `set_data` can be used to preprocess the data. For instance,
        # if `whiten_y` is True, remove the mean of `y`.


    def compute(self, alignment_estimator, test):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        self.alignment_estimator = alignment_estimator
        source_test = test
        target_pred = alignment_estimator.transform(source_test)
        
        # Compute the score of the alignment.
        # TODO : implement an SVC.
        baseline_score = np.mean(score_voxelwise(source_test, target_pred, self.mask))
        aligned_score = np.mean(score_voxelwise(source_test, target_pred, self.mask))

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            baseline_score=baseline_score,
            value=aligned_score,
        )

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return 0

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        
        source = self.alignment_dataset[self.alignment_dataset['subject'] != self.parameters['target_subject'][0]]["path"].to_list()
        target = self.alignment_dataset[self.alignment_dataset['subject'] == self.parameters['target_subject'][0]]["path"].to_list()
        test = self.projected_dataset[self.projected_dataset['subject'] != self.parameters['target_subject'][0]]["path"].to_list()
        
        return dict(
            source=source,
            target=target,
            test=test,
            mask=self.mask,
        )
