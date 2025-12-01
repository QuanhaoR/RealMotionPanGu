import os
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="./conf/", config_name="config_pangu")
def main(cfg):
    pl.seed_everything(cfg.seed)
    assert os.path.exists(cfg.checkpoint), f"Checkpoint {cfg.checkpoint} does not exist"

    datamodule = instantiate(cfg.datamodule.pl_module, test=cfg.submit)
    model = instantiate(cfg.model.pl_module)

    trainer = pl.Trainer(
        logger=False,
        accelerator="npu",
        devices=cfg.npus if not cfg.submit else 1,
        max_epochs=1,
        strategy="ddp_npu",
    )

    if not cfg.submit:
        trainer.validate(model, datamodule, ckpt_path=cfg.checkpoint)
    else:
        trainer.test(model, datamodule, ckpt_path=cfg.checkpoint)


if __name__ == "__main__":
    main()
