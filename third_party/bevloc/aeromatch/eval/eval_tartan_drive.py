# 第三方实现（复现导读）：TartanDrive 测试入口：加载公开权重，按时序运行 BEVLoc 并导出轨迹。
# Third Party
# Initialize GDAL before torchvision/Pillow, which may load an older libjpeg.
from osgeo import gdal
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

# In House
from aeromatch.config.settings import read_settings_file
from aeromatch.data.load_tartandrive2_dataset import TartanDrive
from aeromatch.models.bev_loc import BEVLoc

# Entrypoint:
if __name__ == "__main__":
    # Create argument parser
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser("TartanDrive Trainer", None, "TartanDriver Trainer for BEVLoc")
    parser.add_argument("-s", "--settings", default="aeromatch/config/predator.json")
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="Pretrained checkpoint matching the settings architecture")
    parser.add_argument("--trajectory", help="Evaluate this single prepared trajectory")
    parser.add_argument("--max-batches", type=int,
                        help="Limit inference for a smoke run; omit for the full trajectory")
    parser.add_argument("--output-dir", type=Path, help="Directory for this run's outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not os.path.isfile(args.checkpoint):
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    # Get settings for dataset
    tartan_drive_settings = read_settings_file(args.settings)
    pl.seed_everything(args.seed, workers=True)
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tartan_drive_settings["out_path"] = str(args.output_dir)
    
    #* Create BEVLoc and pass in all the relevant modules
    model_list = [args.checkpoint]
    for mc in model_list:
        os.environ["BEVLOC_FULL_CHECKPOINT"] = "1"
        model = BEVLoc.load_from_checkpoint(mc, tartan_drive_settings=tartan_drive_settings,
                                           strict=True, weights_only=False)
        model.eval()
        print(f"Strict checkpoint load passed: {len(model.state_dict())} parameter/buffer entries")
        if args.output_dir:
            model.metric_calc.traj_saver.out_dir = str(args.output_dir / "traj")

        #* Parse a file to what is wanted for our PyTorch dataset
        test_dir  = tartan_drive_settings["training"]["val_split"]
        test_traj_dirs = ([args.trajectory] if args.trajectory else
                          [f"{test_dir}/{fp}" for fp in sorted(os.listdir(test_dir))
                           if os.path.isdir(f"{test_dir}/{fp}")])
        print(test_traj_dirs)
        for test_traj_dir in test_traj_dirs:
            test_drive_dataset   = TartanDrive([test_traj_dir], model.get_aero_bev(), tartan_drive_settings)
            print(f"Number of Testing Trajectories: {len(test_traj_dirs)}")

            # Path specifics
            checkpoint_out = tartan_drive_settings["checkpoint_out_path"]
            
            # Model string
            mc_str = mc.split("/")[-1].split(".")[0]
            traj_stem = test_traj_dir.split("/")[-1]
            print(f"{mc_str}_{traj_stem}")
            model.model = f"{mc_str}_{traj_stem}"

            # Test/Evaluate on TartanDrive
            traj_loader = DataLoader(test_drive_dataset,
                                    batch_size=tartan_drive_settings["training"]["batch_size"],
                                    shuffle=False, num_workers=0)
            trainer = pl.Trainer(logger=False, enable_checkpointing=False,
                                 accelerator="gpu", devices=1,
                                 limit_test_batches=args.max_batches or 1.0)
            trainer.test(model=model, dataloaders=traj_loader)
            torch.cuda.empty_cache()
