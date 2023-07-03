from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import ibc_api.utils as ibc
    from nilearn import image
    import os
    from nilearn.maskers import NiftiMasker


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "IBC"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'random_state': [27],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        root = os.path.join('/storage/store/work/haggarwa/', 'benchopt_sprint', 'fmralign_benchopt')
        ibc.authenticate()

        db = ibc.get_info(data_type="statistic_map")
        filtered_db = ibc.filter_data(db, subject_list=["sub-08", "sub-04", "sub-05", "sub-15"], task_list=['RSVPLanguage', 'ArchiStandard'])                                                                             
        downloaded_db = ibc.download_data(filtered_db)
        file_names = downloaded_db['local_path'].str.split('/', expand=False).str[-1]
        file_names = file_names.str.split('_', expand=True)
        file_names = file_names.rename(columns={0: 'subject', 1: 'session', 2: 'task'})
        downloaded_db = downloaded_db.join(file_names)
        mask_file = os.path.join(root, 'ibc_data', 'masks', 'gm_mask_3mm.nii.gz')
        masker = NiftiMasker(mask_img=mask_file, standardize=True).fit()

        return dict(dataset=downloaded_db, mask=masker)
