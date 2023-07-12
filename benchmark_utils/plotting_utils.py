# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%
# get all the parquet files
pqt = pd.read_parquet(
    ("/storage/store3/work/pbarbara/fmralign_benchopt/"
     "outputs/benchopt_run_2023-07-07_18h04m34.parquet")
)

# %%
# creat a long dataframe with all the results
d = {"target": [], "source": [], "solver": [], "accuracy": []}

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
targets = [
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

solvers = [
    "scaled_orthogonal",
    "optimal_transport",
    "ridge_cv",
    "fastsrm",
    "identity",
]

for target in targets:
    for source in source_subjects:
        for solver in solvers:
            pqt_solver = pqt[
                pqt["solver_name"] == f"fmralign[method={solver}]"
            ].reset_index(drop=True)
            pqt_solver.set_index("objective_name", inplace=True)
            d["target"].append(target)
            d["source"].append(source)
            d["solver"].append(solver)
            d["accuracy"].append(
                pqt_solver.at[
                    f"frmalign-benchopt[target_subject={target}]",
                    f"objective_{source}",
                ]
            )
df = pd.DataFrame(data=d)

# Center data around identity
# df['accuracy'] -= df[df['solver'] == 'identity']['accuracy'].mean()

# seaborn plotting
# %%
plt.rcParams["figure.dpi"] = 300
ax = sns.boxplot(
    data=df,
    y="solver",
    x="accuracy",
    color="white",
    showfliers=False,
)
# Add in points to show each observation
sns.stripplot(
    x="accuracy",
    y="solver",
    data=df,
    size=4,
    hue="source",
    dodge=True,
    jitter=False,
    palette="tab10",
)
plt.xlabel("Accuracy")
plt.ylabel("Solver")
plt.legend(
    title="Left-out subject", loc="center left", bbox_to_anchor=(1, 0.5)
)
# Fill with grey rectangles
for i in range(len(solvers)):
    ax.add_patch(
        plt.Rectangle(
            (0, i - 0.5),
            1,
            1,
            fill=True,
            color="grey",
            alpha=0.1 * (1 - i % 2),
        )
    )
for x in np.arange(0.25, 0.45, 0.05):
    plt.axvline(x=x, color="black", alpha=0.2, linestyle="--")
plt.yticks(
    np.arange(len(solvers)),
    [
        "Scaled orthogonal",
        "Optimal transport",
        "Ridge regression",
        "FastSRM",
        "Identity",
    ],
)
plt.title("Mean prediction accuracy over all target subjects")
plt.show()

# %%
sns.scatterplot(
    data=df,
    x="target",
    y="source",
    hue="accuracy",
    size="solver",
    sizes=(10, 100),
)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# %%
sns.barplot(data=df, x="source", y="accuracy", hue="solver", ci="sd")
# %%
sns.kdeplot(data=df, x="accuracy", hue="solver", multiple="stack", bins=20)
# %%
