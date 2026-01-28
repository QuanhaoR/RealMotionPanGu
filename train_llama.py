import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate

import time
from pathlib import Path
import torch
import torch_npu

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config_llama")
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
    
 #load llama checkpoint
    print("DEBUG: Checking model structure...")
    print(f"hasattr(model, 'model'): {hasattr(model, 'model')}")
    if hasattr(model, 'model'):
        print(f"hasattr(model.model, 'llama'): {hasattr(model.model, 'llama')}")
        print(f"model.model.llama type: {type(model.model.llama)}")
        print(f"hasattr(cfg, 'llama_path'): {hasattr(cfg, 'llama_path')}")
        if hasattr(cfg, 'llama_path'):
            print(f"cfg.llama_path: {cfg.llama_path}")
            
            
    from transformers import AutoModel

    if (hasattr(model, 'model') and hasattr(model.model, 'llama') and hasattr(cfg, 'llama_path')):
        print("Loading LLaMA using HuggingFace from_pretrained...")
        start_time = time.time()
        
        try:
            hf_model = AutoModel.from_pretrained(cfg.llama_path)
            checkpoint = hf_model.state_dict()
            print(f"Successfully loaded {len(checkpoint)} parameters")
            
            model.model.llama.custom_load_state_dict(checkpoint, tail=True, strict=False)
            
            del hf_model
            print(f"Loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"HuggingFace loading failed: {e}")
            
            
    if (hasattr(model, 'model') and hasattr(model.model, 'llama') and hasattr(cfg, 'llama_path') and cfg.llama_path): 
        print("Loading LLaMA checkpoints")
        start_time = time.time()
        
        llama_path = Path(cfg.llama_path)
        
        bin_files = sorted(llama_path.glob("pytorch_model-*.bin"))
        if not bin_files:
            bin_files = list(llama_path.glob("pytorch_model.bin"))
        
        if bin_files:
            print(f"Found {len(bin_files)} checkpoint files")
            
            combined_checkpoint = {}
            for ckpt_path in bin_files:
                print(f"Loading {ckpt_path.name}")
                checkpoint_part = torch.load(ckpt_path, map_location="cpu")
                combined_checkpoint.update(checkpoint_part)
            
            if hasattr(model.model.llama, 'custom_load_state_dict'):
                model.model.llama.custom_load_state_dict(combined_checkpoint, tail=True, strict=True)
                print(f"Loaded {len(bin_files)} files in {time.time() - start_time:.2f} seconds")
                
                print("LLaMA parameters frozen")
            else:
                print("LLaMA module doesn't have custom_load_state_dict method")
        else:
            print(f"No LLaMA checkpoint files found in {cfg.llama_path}")
    else:
        print("Skipping LLaMA weight loading")
        if hasattr(model, 'model'):
            print(f"model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
    
    #logger.info(model)
    device = torch.device(f"npu:{torch.npu.current_device()}")
    model = model.to(device)
    print('\n'+'-'*50)
    print(f'model device:{model.device}')
    print('\n'+'-'*50)
    print(model)

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
