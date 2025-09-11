import wandb
import pathlib
import pdb
# run = wandb.init()
# artifact = run.use_artifact('maheswararao-university-of-wisconsin-madison/sngp_core/model-tq26lk4l:v0', type='model')
# artifact_dir = artifact.download()

PROJECT_NAME='sngp_core'
MODEL_NAME='model-tq26lk4l'

run = wandb.init(project=PROJECT_NAME, job_type="inference")

# latest model artifact
model_at    = run.use_artifact(MODEL_NAME + ":latest")

# directory where the artifact files are materialised
artifact_dir = pathlib.Path(model_at.download())
print("artifact directory:", artifact_dir)

# file we added earlier with add_file() during training
model_path = artifact_dir

pdb.set_trace()