import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate

import time
from pathlib import Path
import torch
from safetensors.torch import load_file  

import torch_npu  # import NPU support
#from lightning_npu.strategies.npu_parallel import NPUParallelStrategy
#from lightning_npu.accelerators.npu import NPUAccelerator


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config_pangu")
def main(cfg):
    logger.info(f"Experiments are stored in {cfg.output_dir}")
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    
    
    # check NPU
    if torch.npu.is_available():
        device_count = torch.npu.device_count()
        logger.info(f"Found {device_count} NPU devices")
        for i in range(device_count):
            logger.info(f"NPU {i}: {torch.npu.get_device_name(i)}")
    else:
        logger.warning("NPU not available")


    datamodule = instantiate(cfg.datamodule.pl_module, logger=logger)

    model = instantiate(cfg.model.pl_module)
    print("DEBUG: Checking model structure...")
    print(f"hasattr(model, 'model'): {hasattr(model, 'model')}")
    if hasattr(model, 'model'):
        print(f"hasattr(model.model, 'pangu'): {hasattr(model.model, 'pangu')}")
        print(f"model.model.pangu type: {type(model.model.pangu)}")
        print(f"hasattr(cfg, 'pangu_path'): {hasattr(cfg, 'pangu_path')}")
        if hasattr(cfg, 'pangu_path'):
            print(f"cfg.pangu_path: {cfg.pangu_path}")
    if (hasattr(model, 'model') and hasattr (model.model,'pangu') and hasattr(cfg, 'pangu_path')):
        print("Loading PanGu checkpoints")
        start_time = time.time()
        
        pangu_path = Path(cfg.pangu_path)
        
        safetensors_files = list(pangu_path.glob("*.safetensors"))
        bin_files = list(pangu_path.glob("*.bin")) 
        
        if safetensors_files:
            ckpt_path = safetensors_files[0]
            print(f"Loading from safetensors: {ckpt_path.name}")
            checkpoint = load_file(ckpt_path, device="cpu")
        elif bin_files:
            ckpt_path = bin_files[0]
            print(f"Loading from bin: {ckpt_path.name}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        else:
            print(f"No checkpoint files found in {cfg.pangu_path}")
            checkpoint = None
        
        if checkpoint is not None:
            if hasattr(model.model.pangu, 'custom_load_state_dict'):
                model.model.pangu.custom_load_state_dict(checkpoint, strict=True)
            elif hasattr(model.model.pangu, 'load_state_dict'):
                model.model.pangu.load_state_dict(checkpoint, strict=True)
            else:
                print("PanGu module doesn't have expected loading method")
            
            print(f"Loaded in {time.time() - start_time:.2f} seconds")
        else:
            print('No valid checkpoint found') 
    else:
        print("Skipping PanGu weight loading - model structure not compatible")
        if hasattr(model, 'model'):
            print(f"model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
    
    #logger.info(model)

    callbacks = instantiate(cfg.callbacks)
    
    #set Trainer using NPU
    trainer_config = dict(cfg.trainer)
    
    #ensure using NPU
    trainer_config.update({
        "accelerator":"npu",#NPUAccelerator(),
        "devices": cfg.npus,  # using config NPU nums
        "strategy":'ddp_npu',#NPUParallelStrategy(),
    })
    
    logger.info(f"Training on {cfg.npus} NPUs")
    trainer = pl.Trainer(
        callbacks=callbacks,
        **trainer_config
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
    trainer.validate(model, datamodule.val_dataloader())


if __name__ == "__main__":
    main()