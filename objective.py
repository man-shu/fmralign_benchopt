from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from fmralign.metrics import score_voxelwise
    from nilearn.image import concat_imgs


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
        'target_subject': ['sub-01', 'sub-04', 'sub-05', 'sub-06'],
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
        self.source = self.alignment_dataset[self.alignment_dataset['subject'] != self.parameters['target_subject'][0]]["path"].to_list()
        self.target = self.alignment_dataset[self.alignment_dataset['subject'] == self.parameters['target_subject'][0]]["path"].to_list()
        test_dataset = self.projected_dataset[self.projected_dataset['subject'] != self.parameters['target_subject'][0]]["path"].to_list()
        # Concatenate the images to have a single 4D image.
        print("Concatenating the images\n")
        self.test_dataset = concat_imgs(test_dataset)        
        print("Done concatenating the images\n")
        self.mask = mask

        # `set_data` can be used to preprocess the data. For instance,
        # if `whiten_y` is True, remove the mean of `y`.


    def compute(self, alignment_estimator):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        
        source_test = self.test_dataset
        target_pred = alignment_estimator.transform(source_test)
        
        # Compute the score of the alignment.
        # TODO : implement an SVC.
        baseline_score = np.mean(score_voxelwise(source_test, target_pred, self.mask, loss='corr'))
        aligned_score = np.mean(score_voxelwise(source_test, target_pred, self.mask, loss='corr'))

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
        
        return dict(
            source=self.source,
            target=self.target,
            mask=self.mask,
        )
