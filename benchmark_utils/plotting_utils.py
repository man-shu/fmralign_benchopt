# %%
import pandas as pd
import numpy as np

pqt = pd.read_parquet("outputs/benchopt_run_2023-07-04_17h26m33.parquet")
pqt

# %%
source_subjects = [
    "sub-01",
    "sub-04",
    "sub-05",
    "sub-06",
    "sub-07",
    "sub-09",
    "sub-11",
    "sub-13",
    "sub-14",
]
targets = ["sub-01", "sub-04", "sub-05"]

res = np.zeros((len(targets), len(source_subjects)))

for i, target in enumerate(targets):
    for j, source in enumerate(source_subjects):
        res[i, j] = pqt.at[i, f"objective_{source}"]

print(res)

# %%
import matplotlib.pyplot as plt

for i, target in enumerate(targets):
    plt.scatter(res[i, :], [i] * res.shape[1], label=target)
plt.xlabel("Decoding accuracy")
plt.xlim(0, 1)
plt.yticks(range(len(targets)), targets)
plt.legend()
plt.show()

# %%
plt.imshow(res, cmap="magma")
plt.xticks(range(len(source_subjects)), source_subjects, rotation=45)
plt.yticks(range(len(targets)), targets)
plt.xlabel("Left-out subject")
plt.ylabel("Target subject")
plt.colorbar()
plt.show()
