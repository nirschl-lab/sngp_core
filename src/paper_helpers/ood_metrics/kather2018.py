from src.metrics.auc import AUROC_across_dataset
import pandas as pd

id_names = ['kather2018']
ood_names = ['kather2016', 'nirschl', 'wong', 'tang', 'jung', 'acevedo']

wong_file_names = {
    'nirschl':'nirschl_et_al_2018.csv', 
    'kather2016':'kather_et_al_2016.csv',
    'acevedo':'acevedo_et_al_2020.csv',
    'kather2018': 'kather_et_al_2018.csv',
    'jung': 'jung_et_al_2022.csv',
    'tang': 'tang_et_al_2019.csv',
    'wong': 'wong_et_al_2022.csv'
}

baseline_csv_path = 'csv/final/kather2018_baseline'
montae_carlo_csv_path = 'csv/final/kather2018_mc'
sngp_csv_path = 'csv/final/kather2018_sngp'

csv_data = []

# print("----- Baseline Results -----")
csv_path = baseline_csv_path
res = AUROC_across_dataset(csv_path, wong_file_names, id_names, ood_names)
# print(res)
# Add method name as first column
row_data = {'Method': 'Baseline'}
row_data.update(res)
csv_data.append(row_data)

# print("----- MC Dropout Results -----")
csv_path = montae_carlo_csv_path
res = AUROC_across_dataset(csv_path, wong_file_names, id_names, ood_names)
# print(res)
row_data = {'Method': 'MC Dropout'}
row_data.update(res)
csv_data.append(row_data)

# print("----- SNGP Results -----")
csv_path = sngp_csv_path
res = AUROC_across_dataset(csv_path, wong_file_names, id_names, ood_names)
# print(res)
row_data = {'Method': 'SNGP'}
row_data.update(res)
csv_data.append(row_data)

# Create DataFrame and save to CSV
df = pd.DataFrame(csv_data)
df.to_csv('csv/final/wong_results.csv', index=False)
print("\nResults saved to csv/final/wong_results.csv")
print(df)