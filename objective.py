from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    from nilearn.image import concat_imgs
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import normalize
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.base import clone
    from tqdm import tqdm
    from nilearn.image import concat_imgs
    from joblib import Parallel, delayed


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
        "target_subject": ["sub-01", 
                           "sub-04", 
                           "sub-05",
                           "sub-06",
                           "sub-07",
                           "sub-09",
                           "sub-11",
                           "sub-13",
                           "sub-14",
                           ],
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
        self.source_subjects = (
            self.alignment_dataset[
                self.alignment_dataset["subject"] != self.target_subject
            ]["subject"]
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
        self.mask = mask

        # `set_data` can be used to preprocess the data. For instance,
        # if `whiten_y` is True, remove the mean of `y`.

    def compute(self, output_dic):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        alignment_estimators = output_dic["alignment_estimators"]
        method = output_dic["method"]
        pipeline = make_pipeline(LinearSVC(max_iter=10000))
        groups = self.projected_dataset[
            self.projected_dataset["subject"] != self.target_subject
        ]["subject"].values
        X = []
        for subject in self.source_subjects:
            if method == "fastsrm" :
                input_data = self.mask.transform(
                            concat_imgs(self.projected_dataset[self.projected_dataset["subject"] == subject]["path"].unique().tolist())
                        )
                X.append(
                    alignment_estimators[subject].transform(
                        [input_data.T]
                    ).T
                )
            else:
                X.append(
                    self.mask.transform(alignment_estimators[subject].transform(
                            self.projected_dataset[self.projected_dataset["subject"] == subject]["path"].unique().tolist()
                        )
                    )
                )
        X = np.vstack(X)
        X = normalize(X, axis=1)
        y = self.projected_dataset[
            self.projected_dataset["subject"] != self.target_subject
        ]["condition"].values
        # Leave one subject out cross validation
        dict_acc = {}
        logo = LeaveOneGroupOut()
        print("Starting cross validation\n")
        
        def decode(train, test):
            clf = clone(pipeline)
            clf.fit(X[train], y[train])
            score = clf.score(X[test], y[test])
            source_sub = groups[test][0]
            return score, source_sub
        
        score_list = Parallel(n_jobs=10, verbose=3)(delayed(decode)(train, test) for train, test in logo.split(X, y, groups=groups))
        
        for score, source_sub in score_list:
            dict_acc[source_sub] = score

        dict_acc["value"] = np.mean(list(dict_acc.values()))
        
        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict_acc

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
            source_subjects=self.source_subjects,
            target=self.target,
            mask=self.mask,
        )
