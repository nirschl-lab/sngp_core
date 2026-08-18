import torch
import pdb
from src.models.sngp_lit_module import SNGPLitModule
from src.models.sngp_models import SNGPCustom
from src.fileio.hdf.readers import load_dataset
from lightning import LightningDataModule, LightningModule, Trainer


input_dim = 384
num_classes = 8
batch_size = 4
# net = BaselineModel(input_dim, num_classes)
net = SNGPCustom(input_dim, num_classes)
module = SNGPLitModule(net, None, None, False, num_classes=num_classes)

ckpt_path = 'artifacts/model-tq26lk4l:v0/model.ckpt'

features = 'dinov2_vit_S14'
feature_cache = '/data1/shared/cache/feature_cache/'

dataset = 'acevedo_et_al_2020'
# dataset = 'tang_et_al_2019'

train_df = load_dataset(dataset, features, split = 'train', feature_cache = feature_cache)
val_df = load_dataset(dataset, features, split = 'validation', feature_cache = feature_cache)
test_df = load_dataset(dataset, features, split = 'test', feature_cache = feature_cache)

X_train = torch.tensor(train_df[train_df.columns[8:]].to_numpy(), dtype=torch.float32)
y_train = torch.tensor(train_df["label"].values, dtype=torch.long)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=64, shuffle=False)

trainer = Trainer(accelerator="auto", devices=1)  # adjust devices as needed

# Use test_loader or any DataLoader for prediction
predictions = trainer.predict(model=module, dataloaders=train_loader) # get a list of batch prediction

pdb.set_trace()
# predictions will be a list of outputs from each batch's predict_step
# You can concatenate or process as needed:
import torch
all_preds = torch.cat([out['preds'] for out in predictions])
all_probs = torch.cat([out['probs'] for out in predictions])
all_targets = torch.cat([out['targets'] for out in predictions])


print("Predictions:", all_preds)
print("Probabilities:", all_probs)
print("Targets:", all_targets)