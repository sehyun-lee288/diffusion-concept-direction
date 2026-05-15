# 01 — Random-anchor sign-pattern boundary

Phase 4–5 · `scripts/04_compute_sign_grid.py`, `scripts/05_visualize.py` ·
figure `figures/exp1_boundary.png`

## 실험 목적
DDPM CelebA-HQ의 U-Net bottleneck (`mid_block` 출력) 위에서 RDR/E-GBAS
스타일의 **sign-pattern decision boundary 시각화**가 가능한지 확인.

## 왜 하는지
세 논문(E-GBAS, SplineCam, RDR)이 GAN·classifier에서 보여준 paradigm을
diffusion으로 transfer할 수 있는지가 본 연구의 출발점. h-space는
Asyrp이 semantic이라고 주장한 공간이므로 그 안의 boundary 구조가
의미를 가질 가능성이 있음.

## 가설
1. 세 anchor (real CelebA-HQ 이미지의 q-sampled h_500) 가 span하는 2D
   plane 위에서, 채널별 `sign(spatial_mean(h_c))` 가 polytope-like
   region 구조를 만들 것.
2. 같은 region 안의 decoded image는 비슷한 attribute를 가질 것.

## 결과
- Plane 자체는 정확히 구성 (3 점이 affine plane 유일 결정)
- 처음 시도한 spatial-max → 모든 channel이 항상 +1 → 정보 zero (degenerate)
- spatial-mean으로 교체 → 채널간 sign Hamming distance ≈ 215/512 (의미 있음)
- 비-unit orthogonal basis로 anchor가 (0,0), (1,0), (a,1)에 매핑되도록
- 50×50 grid 결과: 2432 unique pattern / 2500 cell — 거의 cell당 unique
- top-K=20 most-balanced channel 사용 시 134 region, 시각적으로
  **방사형(radial)** 패턴이 (0.5, 0.5) 근처에서 만남
- 9개 sparse decoded thumbnail은 모두 유사한 face — h-only inject의 약한
  효과 (Finding 5의 첫 단서)

## Next steps
- 방사형이 모델 특성인지 selection artifact인지 확인 → **doc 02**
- 진짜 neuron 단위 boundary는 어떻게 생겼는지 → **doc 03–04**

## Bonus figure (보충)

`figures/exp13_random_anchor_all_lines.png` — top-K이 아닌 **모든 472개
active channel boundary line**을 한 panel에 plot. 시각적으로 방사형 흔적
없음을 직접 확인 (doc 02 결론의 visual 형태).

