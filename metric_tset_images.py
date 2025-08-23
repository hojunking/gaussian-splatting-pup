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
# 이 스크립트를 실행하는 위치에서 utils 폴더에 접근 가능해야 합니다.
try:
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
except ImportError:
    print("Error: Could not import 'ssim' or 'psnr'.")
    print("Please ensure that the 'utils' directory from the 3D Gaussian Splatting project is accessible.")
    exit()

def compare_test_sets(qp_exp_path: Path):
    """
    압축 모델과 원본 모델 experiment 폴더 내의 test set GT 이미지들을 비교합니다.
    """
    print(f"\nProcessing experiment: {qp_exp_path.name}")

    # 1. 원본 experiment 경로 추론
    exp_name = qp_exp_path.name
    if "_qp" not in exp_name:
        print(f"Error: '{exp_name}' is not a QP experiment directory. Skipping.")
        return

    qp_index = exp_name.find("_qp")
    original_exp_name = exp_name[:qp_index]
    original_exp_path = qp_exp_path.parent / original_exp_name

    if not original_exp_path.exists():
        print(f"Error: Original experiment path not found at {original_exp_path}")
        return

    qp_test_dir = qp_exp_path / "test"
    original_test_dir = original_exp_path / "test"

    if not qp_test_dir.exists() or not original_test_dir.exists():
        print("Error: 'test' directory not found in one or both experiment paths.")
        return
    
    all_avg_results = {}
    all_per_view_results = {}

    # 'test' 폴더 안의 method 별로 (e.g., ours_7000, ours_30000) 모두 비교
    for method_name in sorted(os.listdir(qp_test_dir)):
        # .ipynb_checkpoints와 같은 숨김 파일을 건너뜁니다.
        if method_name.startswith('.'):
            continue

        print(f"\n-- Comparing method: {method_name} --")
        original_gt_dir = original_test_dir / method_name / "gt"
        qp_gt_dir = qp_test_dir / method_name / "gt"

        print(f" -> Original GT (Reference): {original_gt_dir}")
        print(f" -> Compressed GT (Target):  {qp_gt_dir}")

        if not original_gt_dir.exists() or not qp_gt_dir.exists():
            print(f"Warning: 'gt' folder for method '{method_name}' not found. Skipping.")
            continue

        # 2. 이미지 불러오기
        original_images, qp_images, image_names = [], [], []
        for fname in sorted(os.listdir(qp_gt_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                orig_img_path = original_gt_dir / fname
                if orig_img_path.exists():
                    orig_img = Image.open(orig_img_path)
                    qp_img = Image.open(qp_gt_dir / fname)
                    
                    original_images.append(tf.to_tensor(orig_img).unsqueeze(0)[:, :3, :, :].cuda())
                    qp_images.append(tf.to_tensor(qp_img).unsqueeze(0)[:, :3, :, :].cuda())
                    image_names.append(fname)
        
        if not image_names:
            print(f"No matching images found for method '{method_name}'.")
            continue

        # 3. 메트릭 측정
        ssims, psnrs, lpipss = [], [], []
        for i in tqdm(range(len(original_images)), desc=f"Comparing GTs for {method_name}"):
            ssims.append(ssim(qp_images[i], original_images[i]))
            psnrs.append(psnr(qp_images[i], original_images[i]))
            lpipss.append(lpips(qp_images[i], original_images[i], net_type='vgg'))

        # 4. 결과 집계
        avg_results = {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item()
        }
        per_view_results = {
            "SSIM": {name: s.item() for s, name in zip(ssims, image_names)},
            "PSNR": {name: p.item() for p, name in zip(psnrs, image_names)},
            "LPIPS": {name: l.item() for l, name in zip(lpipss, image_names)}
        }
        
        all_avg_results[method_name] = avg_results
        all_per_view_results[method_name] = per_view_results

        print(f"\n[Average Metrics for {method_name}]")
        print(f"  SSIM : {avg_results['SSIM']:.7f}")
        print(f"  PSNR : {avg_results['PSNR']:.7f}")
        print(f"  LPIPS: {avg_results['LPIPS']:.7f}")

    # 5. 최종 JSON 파일 저장
    results_path = qp_exp_path / "testset_quality_results.json"
    per_view_path = qp_exp_path / "testset_quality_per_view.json"

    with open(results_path, 'w') as f:
        json.dump(all_avg_results, f, indent=4)
    with open(per_view_path, 'w') as f:
        json.dump(all_per_view_results, f, indent=4)

    print(f"\n\nFinal results for all methods saved to: {results_path}")
    print(f"Final per-view results saved to: {per_view_path}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f"Using device: {device}")

    parser = ArgumentParser(description="Compare test set GT images between original and compressed experiments.")
    parser.add_argument('--qp_exp_paths', '-p', required=True, nargs="+", type=str,
                        help="Path(s) to the compressed experiment directory (e.g., .../experiments/scene0000_00_qp27).")
    args = parser.parse_args()

    for path_str in args.qp_exp_paths:
        compare_test_sets(Path(path_str))