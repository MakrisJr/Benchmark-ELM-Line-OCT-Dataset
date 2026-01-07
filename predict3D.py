import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from dataset import D3Dataset
from model import UNet3DFrawley, UNet2DEnc3DDec, UNet3D, UNet3D_Aniso

# -----------------------------
# Dice helpers
# -----------------------------
def dice_from_binary(pred: torch.Tensor, gt: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    pred, gt: binary tensors with shape [B, 1, D, H, W] (0/1)
    returns: dice per item in batch shape [B]
    """
    pred = pred.float()
    gt = gt.float()
    dims = (1, 2, 3, 4)
    inter = (pred * gt).sum(dim=dims)
    p = pred.sum(dim=dims)
    g = gt.sum(dim=dims)
    return (2.0 * inter + smooth) / (p + g + smooth)

def match_depth(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Ensure pred and gt have same depth by center-cropping or padding pred.
    pred, gt: [B,1,D,H,W]
    """
    dp = pred.shape[2]
    dg = gt.shape[2]
    if dp == dg:
        return pred

    if dp > dg:
        # center crop
        start = (dp - dg) // 2
        return pred[:, :, start:start+dg, :, :]
    else:
        # pad equally on both sides
        pad_total = dg - dp
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        # pad format for 5D is (W_left, W_right, H_left, H_right, D_left, D_right)
        return torch.nn.functional.pad(pred, (0, 0, 0, 0, pad_left, pad_right), mode="constant", value=0)

def save_volume_slices(pred_bin: np.ndarray, out_dir: str, eye_id: str):
    """
    pred_bin: (D,H,W) uint8 in {0,1}
    saves D slices as pngs 0..D-1
    """
    os.makedirs(out_dir, exist_ok=True)
    D = pred_bin.shape[0]
    for z in range(D):
        out_path = os.path.join(out_dir, f"{eye_id}-{z}.png")
        cv2.imwrite(out_path, (pred_bin[z] * 255).astype(np.uint8))

# -----------------------------
# Main evaluation
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth state_dict")
    parser.add_argument("--model", type=str, default="UNet3DFrawley", help="Model class name selector")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./eval-3d-outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Construct model
    # -----------------------------
    # Add more options if you want to select among models.
    if args.model == "UNet3DFrawley":
        net = UNet3DFrawley(in_channels=1, out_channels=1)
    elif args.model == "UNet2DEnc3DDec":
        net = UNet2DEnc3DDec(in_channels=1, out_channels=1)
    elif args.model == "UNet3D":
        net = UNet3D(in_channels=1, out_channels=1)
    elif args.model == "UNet3D_Aniso":
        net = UNet3D_Aniso(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.to(device)
    net.eval()

    # -----------------------------
    # Dataset / loader (TEST)
    # -----------------------------
    test_img_dir = os.path.join(args.base_dir, "data/test/image/")
    test_mask_dir = os.path.join(args.base_dir, "data/test/mask/")
    test_dataset = D3Dataset(test_img_dir, test_mask_dir, scale=1, transform=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # -----------------------------
    # Dice accumulators
    # -----------------------------
    dice_list = []

    total_intersection = 0.0
    total_pred_sum = 0.0
    total_gt_sum = 0.0
    smooth = 1e-7

    # To map batch index -> eye_id for saving, reuse dataset.eye_ids ordering
    eye_ids = test_dataset.eye_ids

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs = batch["image"].to(device=device, dtype=torch.float32)  # [B,1,D,H,W]
            gts  = batch["mask"].to(device=device, dtype=torch.float32)   # [B,1,D,H,W] in {0,1} (as per your dataset)
            # shuffle depth order
            # B, C, D, H, W = imgs.shape
            # depth_indices = torch.randperm(D)
            # imgs = imgs[:, :, depth_indices, :, :]
            # gts  = gts[:, :, depth_indices, :, :]   

            logits = net(imgs)                                           # [B,1,D?,H,W]
            probs = torch.sigmoid(logits)
            pred_bin = (probs > args.threshold).to(torch.uint8)

            # depth alignment if needed
            pred_bin = match_depth(pred_bin, gts)

            # ensure GT is binary 0/1 (robust)
            gt_bin = (gts > 0.5).to(torch.uint8)

            # per-volume dice
            d = dice_from_binary(pred_bin, gt_bin, smooth=smooth)         # [B]
            dice_list.extend(d.detach().cpu().tolist())

            # global dice accumulation
            # intersection and sums over all voxels for each batch
            inter = (pred_bin & gt_bin).sum().item()
            p_sum = pred_bin.sum().item()
            g_sum = gt_bin.sum().item()

            total_intersection += inter
            total_pred_sum += p_sum
            total_gt_sum += g_sum

            # optionally save predictions
            if args.save_preds:
                B = pred_bin.shape[0]
                for b in range(B):
                    eye_id = eye_ids[i * args.batch_size + b]
                    vol = pred_bin[b, 0].detach().cpu().numpy().astype(np.uint8)  # (D,H,W)
                    save_volume_slices(vol, args.out_dir, eye_id)

            print(f"[{i+1}/{len(test_loader)}] Dice (batch mean): {float(np.mean(d.detach().cpu().numpy())):.6f}")

    if len(dice_list) == 0:
        print("No volumes evaluated.")
        return

    mean_dice = float(np.mean(dice_list))
    global_dice = (2.0 * total_intersection + smooth) / (total_pred_sum + total_gt_sum + smooth)

    print("\n====================")
    print(f"Volumes evaluated: {len(dice_list)}")
    print(f"Mean Dice (per-volume average): {mean_dice:.6f}")
    print(f"Total Dice (global over all voxels): {global_dice:.6f}")
    print("====================\n")


if __name__ == "__main__":
    main()

"""
python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet3DFrawley_Nov-25-2025_1508_model/checkpoints/UNet3DFrawley_Nov-25-2025_1508_model_best_epoch_66.pth --model UNet3DFrawley --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet2DEnc3DDec_Jan-05-2026_1240_model/checkpoints/UNet2DEnc3DDec_Jan-05-2026_1240_model_best_epoch_85.pth --model UNet2DEnc3DDec --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet3D_Nov-21-2025_1259_model/checkpoints/UNet3D_Nov-21-2025_1259_model_best_epoch_55.pth --model UNet3D --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet3D_Aniso_Nov-25-2025_1449_model/checkpoints/INTERRUPTED_UNet3D_Aniso_Nov-25-2025_1449_model.model --model UNet3D_Aniso --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs
"""