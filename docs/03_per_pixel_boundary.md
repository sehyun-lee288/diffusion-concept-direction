# 03 — Per-pixel boundary (single spatial location)

Phase 6 · `scripts/07_single_location.py` ·
figure `figures/exp2_single_location_pix4_4.png`

## 실험 목적
Channel별 spatial-mean 대신, **고정된 한 픽셀 (i*, j*) 에서의 channel sign**
으로 boundary 그림. 즉 진짜 (channel, spatial) neuron 단위 boundary.

## 왜 하는지
doc 02 결론: top-K-balanced selection이 방사형을 만든다. Reduction 자체를
바꾸면 (spatial-mean → single-pixel) 더 진실에 가까운 model geometry가
보일 것. 각 boundary line이 단일 실제 U-Net neuron의 zero-crossing이 됨.

## 가설
1. (4, 4) 중심 픽셀 단일 위치의 sign vector는 channel 별로 distinct → 더
   풍부한 region 구조
2. Boundary line들의 방향 분포가 **uniform** (방사형 없음) — 진짜 model
   structure를 반영

## 결과
- 475 / 512 channel이 active (window에서 sign 변함)
- 2438 regions / 2500 cells — grid 해상도 한계 (475-bit signature가 거의
  cell당 unique)
- **오른쪽 panel: 475개 line이 plane을 균일하게 분할** — 방사형 흔적 없음
- 왼쪽 region map은 fragmented (해상도 saturation) 하지만 boundary 자체는
  깨끗

## Next steps
- 채널 단위로는? 같은 conv filter가 64 spatial location 적용 → **doc 04**
- region map이 너무 fragmented → 채널 수 줄여야 의미 있는 region 시각화
  (cluster, PCA 등 future)
- 방사형이 model property가 아님은 확인 → "이건 selection 문제일 뿐"
- 그러나 다음 질문: 그래서 의미 있는 region은 어떻게 얻을 수 있나? →
  doc 05–10 사슬
