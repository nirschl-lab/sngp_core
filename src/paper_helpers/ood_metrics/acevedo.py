from src.metrics.auc import AUROC_across_dataset
import pandas as pd

id_names = ['acevedo']
ood_names = ['jung','wong','kather2016', 'kather2018',   'nirschl', 'tang']

acevedo_fid_scores = {'jung':306.65, 
              'wong':351.25, 
              'kather2016':375.57, 
              'kather2018':377.93, 
              'nirschl':379.89,
              'tang':388.94}

acevedo_sampling_populations = {}

acevedo_file_names = {
    'nirschl':'nirschl_et_al_2018.csv', 
    'kather2016':'kather_et_al_2016.csv',
    'acevedo':'acevedo_et_al_2020.csv',
    'kather2018': 'kather_et_al_2018.csv',
    'jung': 'jung_et_al_2022.csv',
    'tang': 'tang_et_al_2019.csv',
    'wong': 'wong_et_al_2022.csv'
}

baseline_csv_path = 'csv/final/acevedo_baseline'
montae_carlo_csv_path = 'csv/final/acevedo_mc'
sngp_csv_path = 'csv/final/acevedo_sngp'

csv_data = []

# print("----- Baseline Results -----")
csv_path = baseline_csv_path
res = AUROC_across_dataset(csv_path, acevedo_file_names, id_names, ood_names)
# print(res)
# Add method name as first column
row_data = {'Method': 'Baseline'}
row_data.update(res)
csv_data.append(row_data)

# print("----- MC Dropout Results -----")
csv_path = montae_carlo_csv_path
res = AUROC_across_dataset(csv_path, acevedo_file_names, id_names, ood_names)
# print(res)
row_data = {'Method': 'MC Dropout'}
row_data.update(res)
csv_data.append(row_data)

# print("----- SNGP Results -----")
csv_path = sngp_csv_path
res = AUROC_across_dataset(csv_path, acevedo_file_names, id_names, ood_names)
# print(res)
row_data = {'Method': 'SNGP'}
row_data.update(res)
csv_data.append(row_data)

# Add fid scores to the dataframe
fid_scores = []
for ood_name in ood_names:
    fid_scores.append(acevedo_fid_scores[ood_name])
row_data = {'Method': 'FID Score'}
for i, ood_name in enumerate(ood_names):
    row_data[ood_name] = f"{fid_scores[i]:.2f}"
csv_data.append(row_data)

# Create DataFrame and save to CSV
df = pd.DataFrame(csv_data)
df.to_csv('csv/final/acevedo_results.csv', index=False)
print("\nResults saved to csv/final/acevedo_results.csv")
print(df)

