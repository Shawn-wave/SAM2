import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# ================================
# 파일 경로 설정 - 여기서 직접 수정하세요
# ================================
IMAGE_PATH = "./data/image/test_block.jpg"  # 입력 이미지 경로
SAM_PATH = "./segment-anything-2"  # SAM2 디렉토리 경로
CHECKPOINT = "sam2_hiera_large.pt"  # 체크포인트 파일명
OUTPUT_BASE_DIR = "./output"  # 출력 디렉토리
# ================================

class ImageClickCoordinates:
    def __init__(self, image, marker_color=(0, 255, 0), id=0):
        self.image = image.copy()
        self.coordinates = []
        self.negative_coordinates = []
        self.marker_color = marker_color
        self.image_name = "Image"+str(id)

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down event
            print(f"Positive click: ({x}, {y})")
            self.coordinates.append((x, y))
            cv2.circle(self.image, (x, y), 5, self.marker_color, -1)
            cv2.imshow(self.image_name, self.image)
        if event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button down event
            print(f"Negative click: ({x}, {y})")
            self.negative_coordinates.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)  # Red for negative
            cv2.imshow(self.image_name, self.image)
        
    def show_image_and_collect_coordinates(self):
        print("Left click: positive points, Right click: negative points, Press any key to finish")
        cv2.imshow(self.image_name, self.image)
        cv2.setMouseCallback(self.image_name, self.click_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.coordinates, self.negative_coordinates

def collect_clicks_from_image(image, id=0):
    """이미지에서 클릭을 수집하는 함수"""
    marker_color = (0, 255, 0)  # Green for positive clicks
    click_collector = ImageClickCoordinates(image, marker_color, id)
    pos_coords, neg_coords = click_collector.show_image_and_collect_coordinates()

    if not pos_coords and not neg_coords:
        print("No clicks detected!")
        return None, None

    # 포지티브와 네거티브 클릭을 합침
    all_coords = pos_coords + neg_coords
    points = np.array(all_coords, dtype=np.float32)
    
    # 라벨 할당 (1: positive, 0: negative)
    pos_labels = np.array([1] * len(pos_coords), np.int32)
    neg_labels = np.array([0] * len(neg_coords), np.int32)
    labels = np.concatenate([pos_labels, neg_labels], axis=0)

    return points, labels

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# bfloat16 설정 (GPU가 지원하는 경우)
if device.type == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 이미지 경로 확인
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at {IMAGE_PATH}")
    print("Please update IMAGE_PATH in the code to point to your image file.")
    exit(1)

# 체크포인트 경로 설정
possible_checkpoint_paths = [
    os.path.join(SAM_PATH, "checkpoints", CHECKPOINT),
    os.path.join("./checkpoints", CHECKPOINT),
    os.path.join("./", CHECKPOINT),
    CHECKPOINT
]

ckpt_path = None
for path in possible_checkpoint_paths:
    if os.path.exists(path):
        ckpt_path = path
        print(f"Found checkpoint at: {ckpt_path}")
        break

if ckpt_path is None:
    print("Error: Checkpoint file not found!")
    print("Please download the checkpoint file:")
    print("1. mkdir -p checkpoints")
    print("2. cd checkpoints") 
    print("3. wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
    exit(1)

# 모델 설정
model_cfg = "sam2_hiera_l.yaml"

# SAM2 이미지 예측기 빌드
sam2_model = build_sam2(model_cfg, ckpt_path, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# 이미지 로드
image = Image.open(IMAGE_PATH)
image = np.array(image.convert("RGB"))
print(f"Image shape: {image.shape}")

# 이미지를 예측기에 설정
predictor.set_image(image)

# 이미지 이름 추출 (확장자 제거)
image_name = os.path.basename(IMAGE_PATH).split(".")[0]

# 출력 디렉토리 설정 - file과 같은 레벨의 output 폴더 사용
output_dir_name = "output_" + image_name
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, output_dir_name)

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 원본 이미지 저장 (참조용)
original_path = os.path.join(OUTPUT_DIR, "original.jpg")
Image.fromarray(image).save(original_path)

print(f"Original image saved to: {original_path}")
print("Starting interactive segmentation...")

object_id = 0
all_masks = []
all_labels = []

while True:
    # 라벨 이름 입력
    label_name = input(f"\nEnter label name for object {object_id} (or 'exit' to quit): ")
    if label_name.lower() == 'exit':
        break
    
    print(f"\nSegmenting object: {label_name}")
    print("Click on the image to select points...")
    
    # 이미지에서 클릭 수집
    points, labels = collect_clicks_from_image(image.copy(), object_id)
    
    if points is None:
        print("No points selected, skipping...")
        continue
    
    print(f"Collected {len(points)} points")
    
    # SAM2로 예측 수행
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )
    
    # 가장 좋은 마스크 선택 (가장 높은 점수)
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    best_score = scores[best_mask_idx]
    
    print(f"Best mask score: {best_score:.3f}")
    print(f"Mask shape: {best_mask.shape}, dtype: {best_mask.dtype}")
    
    # 마스크를 불린 형태로 변환
    if len(best_mask.shape) == 3:
        best_mask = best_mask.squeeze()  # (1, H, W) -> (H, W)
    best_mask = best_mask.astype(bool)  # 불린 형태로 변환
    
    # 마스크 저장
    all_masks.append(best_mask)
    all_labels.append(label_name)
    
    # 개별 마스크 저장 (PNG + numpy)
    mask_path = os.path.join(OUTPUT_DIR, f"mask_{object_id}_{label_name}.png")
    mask_image = (best_mask.astype(np.uint8) * 255)
    Image.fromarray(mask_image).save(mask_path)
    
    # numpy 형태로도 저장
    mask_npy_path = os.path.join(OUTPUT_DIR, f"mask_{object_id}_{label_name}.npy")
    np.save(mask_npy_path, best_mask)  # 불린 형태로 저장
    
    # 마스킹된 원본 이미지 생성 (마스크 영역만 원본 색상으로)
    masked_original = np.zeros_like(image)
    masked_original[best_mask] = image[best_mask]
    masked_original_path = os.path.join(OUTPUT_DIR, f"masked_original_{object_id}_{label_name}.jpg")
    Image.fromarray(masked_original).save(masked_original_path)
    
    # numpy 형태로도 저장
    masked_original_npy_path = os.path.join(OUTPUT_DIR, f"masked_original_{object_id}_{label_name}.npy")
    np.save(masked_original_npy_path, masked_original)
    
    # 마스크 오버레이 이미지 생성 (원본 + 반투명 색상)
    overlay = image.copy().astype(np.float32)
    color = np.random.randint(0, 255, 3)
    overlay[best_mask] = overlay[best_mask] * 0.6 + color * 0.4
    
    overlay_path = os.path.join(OUTPUT_DIR, f"overlay_{object_id}_{label_name}.jpg")
    Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)
    
    print(f"Mask saved to: {mask_path}")
    print(f"Mask numpy saved to: {mask_npy_path}")
    print(f"Masked original saved to: {masked_original_path}")
    print(f"Masked original numpy saved to: {masked_original_npy_path}")
    print(f"Overlay saved to: {overlay_path}")
    
    object_id += 1

# 모든 마스크를 합친 최종 결과 생성
if all_masks:
    print(f"\nGenerating final result with {len(all_masks)} objects...")
    
    # 최종 오버레이 이미지
    final_overlay = image.copy().astype(np.float32)
    
    # 컬러 맵 생성
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_masks)))[:, :3] * 255
    
    for i, (mask, label) in enumerate(zip(all_masks, all_labels)):
        color = colors[i]
        # 마스크가 불린 형태인지 확인
        if mask.dtype != bool:
            mask = mask.astype(bool)
        final_overlay[mask] = final_overlay[mask] * 0.7 + color * 0.3
    
    # 최종 결과 저장
    final_path = os.path.join(OUTPUT_DIR, "final_segmentation.jpg")
    Image.fromarray(final_overlay.astype(np.uint8)).save(final_path)
    
    # 마스크 조합 저장 (각 객체마다 다른 값)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, mask in enumerate(all_masks):
        # 마스크가 불린 형태인지 확인하고 변환
        if mask.dtype != bool:
            mask = mask.astype(bool)
        combined_mask[mask] = i + 1
    
    combined_mask_path = os.path.join(OUTPUT_DIR, "combined_masks.png")
    Image.fromarray(combined_mask * (255 // len(all_masks))).save(combined_mask_path)
    
    # numpy 형태로도 저장
    combined_mask_npy_path = os.path.join(OUTPUT_DIR, "combined_masks.npy")
    np.save(combined_mask_npy_path, combined_mask)
    
    print(f"Final segmentation saved to: {final_path}")
    print(f"Combined masks saved to: {combined_mask_path}")
    print(f"Combined masks numpy saved to: {combined_mask_npy_path}")
    
    # 결과 요약
    print(f"\nSegmentation completed!")
    print(f"Total objects segmented: {len(all_masks)}")
    print(f"Labels: {', '.join(all_labels)}")
    print(f"Results saved in: {OUTPUT_DIR}")

else:
    print("No objects were segmented.")

print("Done!")