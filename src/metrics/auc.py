from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import os

seeds = [42, 1337, 12345, 8675309, 314159, 271828, 20240427, 987654321, 3735928559, 777]
sample_rate = 1000

def AUROC(ID_MSP, OOD_MSP):
    # Convert to numpy arrays
    ID_MSP = np.asarray(ID_MSP)
    OOD_MSP = np.asarray(OOD_MSP)
    
    # Compute uncertainty scores (higher = more likely OOD)
    id_uncertainty = 1 - ID_MSP # should be 0 in ideal case
    ood_uncertainty = 1 - OOD_MSP # should be 1 in ideal case
    
    # Concatenate scores and labels
    y_true = np.concatenate([np.zeros_like(ID_MSP), np.ones_like(OOD_MSP)])  # 0 = ID, 1 = OOD
    y_scores = np.concatenate([id_uncertainty, ood_uncertainty])
    
    # y_true = np.concatenate([np.ones_like(ID_MSP), np.zeros_like(OOD_MSP)])  # 0 = ID, 1 = OOD
    # y_scores = np.concatenate([ID_MSP, OOD_MSP])

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)
    return auroc

# def AUROC_across_dataset(csv_path, csv_file_names, id_names, ood_names):
#     for id_name in id_names:
#         for ood_name in ood_names:
#             id_df = pd.read_csv(os.path.join(csv_path, csv_file_names[id_name]))
#             id_df = id_df[id_df['fold']=='test']
#             ood_df = pd.read_csv(os.path.join(csv_path, csv_file_names[ood_name]))
#             #calcualte mean and std across 10 different seeds
#             auroc_list = []
#             for seed in seeds:
#                 id_samples = id_df.sample(sample_rate, random_state=seed)['prediction_prob_score']
#                 ood_samples = ood_df.sample(sample_rate, random_state=seed)['prediction_prob_score']
#                 auroc_list.append(AUROC(id_samples, ood_samples))
#             print(f"AUROC {id_name} vs {ood_name} - {np.mean(auroc_list):.4f} ± {np.std(auroc_list):.4f}")


def AUROC_across_dataset(csv_path, csv_file_names, id_names, ood_names):
    res = {}
    for id_name in id_names:
        for ood_name in ood_names:
            id_df = pd.read_csv(os.path.join(csv_path, csv_file_names[id_name]))
            id_df = id_df[id_df['fold']=='test']
            ood_df = pd.read_csv(os.path.join(csv_path, csv_file_names[ood_name]))
            #calcualte mean and std across 10 different seeds
            auroc_list = []
            for seed in seeds:
                id_samples = id_df.sample(sample_rate, random_state=seed)['prediction_prob_score']
                ood_samples = ood_df.sample(sample_rate, random_state=seed)['prediction_prob_score']
                auroc_list.append(AUROC(id_samples, ood_samples))
            # res[ood_name] = (np.mean(auroc_list), np.std(auroc_list))
            res[ood_name] = f"{np.mean(auroc_list):.4f} ± {np.std(auroc_list):.4f}"
        # res[id_name] = sub_res
    return res