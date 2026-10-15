"""
SAM 마스크 기반 선수 박스 정밀화 (3부 21.2절, 정밀 분석 모드 · GPU)

추적기 박스를 프롬프트로 SAM 에 넣어 선수 마스크를 얻고,
- 마스크 최하단 픽셀로 발 위치(지면 접점)를 구해 경기장 좌표 정확도를 높이고
  (박스 하단 중앙은 다리를 벌리거나 박스가 헐거우면 어긋난다)
- 마스크로 배경을 지운 유니폼 크롭을 팀 분류(SigLIP2)에 넘긴다.

모델: facebook/sam2.1-hiera-* (승인 불필요) 또는 facebook/sam3 (Hugging Face 접근 승인 필요).
SAM3 의 박스 프롬프트 추적기(Sam3TrackerModel)는 SAM2 와 같은 API 라 모델 이름만 바꾸면 된다.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MaskResult:
    foot: Optional[Tuple[float, float]]   # 원본 픽셀 (마스크가 이상하면 None → 박스 하단 중앙 사용)
    bbox: Optional[Tuple[int, int, int, int]]
    mask: Optional[np.ndarray] = None     # 박스 크기 bool (요청한 경우만)


def mask_foot(mask: np.ndarray, x0: int = 0, y0: int = 0, band: float = 0.04
              ) -> Optional[Tuple[Tuple[float, float], Tuple[int, int, int, int]]]:
    """
    bool 마스크 → (발 위치, 마스크 외접 박스). 발 위치는 마스크 높이의 하단 band 비율 안 픽셀의
    x 평균, y 는 최하단 행. (x0, y0) 는 마스크 좌상단의 원본 좌표.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return None
    top, bottom = int(ys.min()), int(ys.max())
    low = ys >= bottom - max(1, int((bottom - top) * band))
    foot = (float(xs[low].mean()) + x0, float(bottom) + y0)
    return foot, (int(xs.min()) + x0, top + y0, int(xs.max()) + x0, bottom + y0)


def plausible(bbox: Sequence[int], mbox: Sequence[int], area_ratio: float) -> bool:
    """마스크가 박스 속 선수를 제대로 잡았는지: 면적이 너무 작거나 박스 밖으로 크게 번지면 버림"""
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if area_ratio < 0.15:
        return False
    return (mbox[3] >= y1 + 0.6 * h and mbox[3] <= y2 + 0.15 * h
            and mbox[0] >= x1 - 0.3 * w and mbox[2] <= x2 + 0.3 * w)


class SamRefiner:
    """박스 프롬프트 SAM (이미지 모드). 한 프레임의 모든 박스를 한 번에 추론"""

    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda", min_height: int = 24,
                 fallback: Optional[str] = "facebook/sam2.1-hiera-large"):
        """fallback: model_id 를 받을 수 없을 때(SAM3 접근 미승인 등) 대신 쓸 모델"""
        self.model_id, self.device, self.min_height, self.fallback = model_id, device, min_height, fallback
        self.model = self.processor = None

    def _load(self):
        try:
            self._load_model(self.model_id)
        except Exception as e:  # 승인 필요(gated) 모델이거나 네트워크 문제
            if not self.fallback or self.fallback == self.model_id:
                raise
            print(f"⚠️ {self.model_id} 를 불러오지 못해 {self.fallback} 로 대체합니다: {str(e).splitlines()[0][:120]}")
            self.model_id = self.fallback
            self._load_model(self.model_id)

    def _load_model(self, model_id: str):
        import torch
        if "sam3" in model_id:
            from transformers import Sam3TrackerModel as Model, Sam3TrackerProcessor as Processor
        else:
            from transformers import Sam2Model as Model, Sam2Processor as Processor
        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        self.model = Model.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.processor = Processor.from_pretrained(model_id)

    def refine(self, frame_bgr: np.ndarray, boxes: List[Sequence[int]], return_masks: bool = False
               ) -> List[MaskResult]:
        """boxes: [(x1, y1, x2, y2), ...] 원본 픽셀 → 박스마다 MaskResult (작은 박스는 건너뜀)"""
        import torch
        out = [MaskResult(None, None) for _ in boxes]
        idx = [i for i, b in enumerate(boxes) if b[3] - b[1] >= self.min_height]
        if not idx:
            return out
        if self.model is None:
            self._load()
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        prompt = [[[float(v) for v in boxes[i][:4]] for i in idx]]
        inputs = self.processor(images=rgb, input_boxes=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            pred = self.model(pixel_values=inputs["pixel_values"].to(self.model.dtype),
                              input_boxes=inputs["input_boxes"], multimask_output=False)
        # 원본 해상도 업샘플은 GPU 에서 (CPU 로는 박스 주변만 옮김)
        masks = self.processor.post_process_masks(pred.pred_masks.float(), inputs["original_sizes"])[0][:, 0]
        H, W = masks.shape[1:]
        for k, i in enumerate(idx):
            x1, y1, x2, y2 = map(int, boxes[i][:4])
            pad_y, pad_x = int((y2 - y1) * 0.15), int((x2 - x1) * 0.3)
            ya, yb, xa, xb = max(0, y1 - pad_y), min(H, y2 + pad_y), max(0, x1 - pad_x), min(W, x2 + pad_x)
            sub = masks[k, ya:yb, xa:xb].cpu().numpy()
            fm = mask_foot(sub, xa, ya)
            if fm is None:
                continue
            foot, mbox = fm
            box_area = max((x2 - x1) * (y2 - y1), 1)
            if not plausible((x1, y1, x2, y2), mbox, sub.sum() / box_area):
                continue
            m = None
            if return_masks:
                m = np.zeros((y2 - y1, x2 - x1), bool)
                cy, cx = max(0, y1), max(0, x1)
                part = masks[k, cy:min(H, y2), cx:min(W, x2)].cpu().numpy()
                m[cy - y1:cy - y1 + part.shape[0], cx - x1:cx - x1 + part.shape[1]] = part
            out[i] = MaskResult((round(foot[0], 1), round(foot[1], 1)), mbox, m)
        return out

    def unload(self):
        self.model = self.processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
