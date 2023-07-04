from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    from fmralign.metrics import score_voxelwise
    from nilearn.image import concat_imgs, index_img
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import normalize
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.base import clone


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
        "target_subject": ["sub-01", "sub-04", "sub-05"],
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
        self.source = (
            self.alignment_dataset[
                self.alignment_dataset["subject"] != self.target_subject
            ]["path"]
            .unique()
            .tolist()
        )
        self.target = (
            self.alignment_dataset[
                self.alignment_dataset["subject"] == self.target_subject
            ]["path"]
            .unique()
            .tolist()
        )
        rsvp_imgs = (
            self.projected_dataset[
                self.projected_dataset["subject"] != self.target_subject
            ]["path"]
            .unique()
            .tolist()
        )
        # Concatenate the images to have a single 4D image.
        print("Concatenating the images\n")
        print(len(rsvp_imgs))
        self.rsvp_imgs = concat_imgs(rsvp_imgs)
        print("Done concatenating the images\n")
        self.mask = mask

        # `set_data` can be used to preprocess the data. For instance,
        # if `whiten_y` is True, remove the mean of `y`.

    def compute(self, alignment_estimator):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        pipeline = make_pipeline(LinearSVC(max_iter=10000))
        groups = self.projected_dataset[
            self.projected_dataset["subject"] != self.target_subject
        ]["subject"]
        X = self.mask.transform(alignment_estimator.transform(self.rsvp_imgs))
        X = normalize(X, axis=1)
        y = self.projected_dataset[
            self.projected_dataset["subject"] != self.target_subject
        ]["condition"].values
        # Leave one subject out cross validation
        acc = []
        logo = LeaveOneGroupOut()
        print("Starting cross validation\n")
        for train, test in logo.split(X, y, groups=groups):
            clf = clone(pipeline)
            clf.fit(X[train], y[train])
            score = clf.score(X[test], y[test])
            acc.append(score)
            print("Target: ", self.target_subject)
            print("Score: ", score)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=np.mean(acc),
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
