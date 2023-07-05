# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# get all the parquet files
pqt = pd.read_parquet('../../outputs/benchopt_run_2023-07-05_12h16m30.parquet')
pqt_ridge = pd.read_parquet('../../outputs/benchopt_run_2023-07-05_14h07m32.parquet')
pqt = pd.concat([pqt, pqt_ridge], axis=0).reset_index(drop=True)

# %%
# creat a long dataframe with all the results
d = {'target': [], 'source': [], 'solver': [], 'accuracy': []}

source_subjects = ["sub-01", 
                    "sub-04", 
                    "sub-05",
                    "sub-06",
                    "sub-07",
                    "sub-09",
                    "sub-11",
                    "sub-13",
                    "sub-14",
                    ]
targets = ["sub-01", 
            "sub-04", 
            "sub-05",
            "sub-06",
            "sub-07",
            "sub-09",
            "sub-11",
            "sub-13",
            "sub-14",
            ]

solvers = ['scaled_orthogonal', 'optimal_transport', 'ridge_cv']

for target in targets:
    for source in source_subjects:
        for solver in solvers:
            pqt_solver = pqt[pqt['solver_name'] == f'fmralign[method={solver}]'].reset_index(drop=True)
            pqt_solver.set_index('objective_name', inplace=True)
            d['target'].append(target)
            d['source'].append(source)
            d['solver'].append(solver)
            d['accuracy'].append(pqt_solver.at[f'frmalign-benchopt[target_subject={target}]', f'objective_{source}'])
df = pd.DataFrame(data=d)

# seaborn plotting
# %%
sns.boxplot(data=df, x='solver', y='accuracy', hue='target',)

# %%
sns.scatterplot(data=df, x='target', y='source', hue='accuracy', size='solver', sizes=(10, 100))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
sns.barplot(data=df, x='source', y='accuracy', hue='solver', ci='sd')
# %%
sns.kdeplot(data=df, x='accuracy', hue='solver', multiple='stack', bins=20)
# %%
