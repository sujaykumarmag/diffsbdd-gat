import torch
from pytorch_lightning import Trainer
from lightning_modules import LigandPocketDDPM


checkpoint_path = "egnn+att/SE3-cond-full/checkpoints/last.ckpt"
model = LigandPocketDDPM.load_from_checkpoint(checkpoint_path)


model.eval()
torch.set_grad_enabled(False)

trainer = Trainer(accelerator="cpu", devices=1, precision=16)  
model.trainer = trainer

trainer.validate(model)
