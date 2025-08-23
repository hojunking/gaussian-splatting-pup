import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from argparse import ArgumentParser

# 3DGS 프로젝트의 유틸리티 함수들을 import 합니다.
# utils 폴더가 스크립트와 같은 위치나 PYTHONPATH에 있어야 합니다.
from utils.loss_utils import ssim
from utils.image_utils import psnr

def compare_scene_images(qp_scene_path: Path):
    """
    압축된 씬과 원본 씬의 'color' 폴더 내 이미지들을 비교하고 메트릭을 측정합니다.
    """
    print(f"\nProcessing scene: {qp_scene_path.name}")

    # 1. 원본 씬 경로 추론
    scene_name = qp_scene_path.name
    if "_qp" not in scene_name:
        print(f"Warning: '{scene_name}' does not seem to be a QP directory. Skipping.")
        return

    qp_index = scene_name.find("_qp")
    original_scene_name = scene_name[:qp_index]
    original_scene_path = qp_scene_path.parent / original_scene_name

    original_color_dir = original_scene_path / "color"
    qp_color_dir = qp_scene_path / "color"

    print(f" -> Original images path: {original_color_dir}")
    print(f" -> Compressed images path: {qp_color_dir}")

    if not original_color_dir.exists() or not qp_color_dir.exists():
        print("Error: One or both 'color' directories not found. Skipping.")
        return

    # 2. 이미지 불러오기
    original_images = []
    compressed_images = []
    image_names = []
    
    # 압축 폴더의 이미지 목록을 기준으로 루프
    for fname in sorted(os.listdir(qp_color_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            original_img_path = original_color_dir / fname
            compressed_img_path = qp_color_dir / fname

            if original_img_path.exists():
                orig_img = Image.open(original_img_path)
                comp_img = Image.open(compressed_img_path)

                # 이미지를 PyTorch 텐서로 변환 (3채널 RGB로 고정)
                original_images.append(tf.to_tensor(orig_img).unsqueeze(0)[:, :3, :, :].cuda())
                compressed_images.append(tf.to_tensor(comp_img).unsqueeze(0)[:, :3, :, :].cuda())
                image_names.append(fname)

    if not image_names:
        print("No matching images found to compare.")
        return

    # 3. 메트릭 측정
    ssims, psnrs, lpipss = [], [], []

    for i in tqdm(range(len(original_images)), desc="Comparing images"):
        # 원본(gt) vs 압축(render) 비교
        ssims.append(ssim(compressed_images[i], original_images[i]))
        psnrs.append(psnr(compressed_images[i], original_images[i]))
        lpipss.append(lpips(compressed_images[i], original_images[i], net_type='vgg'))

    # 4. 결과 집계 및 저장
    # 평균 결과 (results.json)
    avg_results = {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item()
    }
    
    print("\n[Average Metrics]")
    print(f"  SSIM : {avg_results['SSIM']:.7f}")
    print(f"  PSNR : {avg_results['PSNR']:.7f}")
    print(f"  LPIPS: {avg_results['LPIPS']:.7f}")

    # 개별 이미지 결과 (per_view.json)
    per_view_results = {
        "SSIM": {name: s.item() for s, name in zip(ssims, image_names)},
        "PSNR": {name: p.item() for p, name in zip(psnrs, image_names)},
        "LPIPS": {name: l.item() for l, name in zip(lpipss, image_names)}
    }

    # JSON 파일로 저장
    results_path = qp_scene_path / "quality_results.json"
    per_view_path = qp_scene_path / "quality_per_view.json"

    with open(results_path, 'w') as f:
        json.dump(avg_results, f, indent=4)
    with open(per_view_path, 'w') as f:
        json.dump(per_view_results, f, indent=4)

    print(f"\nResults saved to: {results_path}")
    print(f"Per-view results saved to: {per_view_path}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    parser = ArgumentParser(description="Compare original and compressed image sets from Scannet dataset.")
    parser.add_argument('--qp_paths', '-p', required=True, nargs="+", type=str,
                        help="Path(s) to the compressed scene directory (e.g., .../scene0000_00_qp27).")
    args = parser.parse_args()

    for path_str in args.qp_paths:
        compare_scene_images(Path(path_str))