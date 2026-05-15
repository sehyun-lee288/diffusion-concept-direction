# 08 — Attribute-paired 2D plane (disentangled semantic space) ✓

Phase 13 · `scripts/14_attribute_plane.py` ·
figure `figures/exp9_attribute_plane.png`

## 실험 목적
1D orth axis가 부분적 decoupling만 (doc 07). 두 attribute axis를 **plane으로
동시에 manipulate** 했을 때 어떻게 보이는지.

## 왜 하는지
- 1D sweep에선 smile_orth만 움직이고 gender는 학습된 noise로만 변화
- 2D plane에선 `α·Δh_smile_orth + β·Δh_gender` 로 둘 다 명시적 control
- 모델이 두 axis를 *동시에* 표현할 수 있다면 disentanglement가 더 깨끗할 것

## 가설
1. 5×5 grid over (α, β) ∈ [−2.5, 2.5]² 각 점에서 multi-step inject
2. 동일 x_T (seed=0) 로 시작 → grid 모든 cell이 같은 identity의 다른 face
3. α 축에서 smile, β 축에서 gender가 각각 monotonic
4. 4 corner가 의도된 attribute 조합 ((smile/no-smile) × (male/female))

## 결과 — **CLEAN DISENTANGLEMENT**

`figures/exp9_attribute_plane.png` 5×5 thumbnail grid:

|        | α = −2.5 | α = 0 | α = +2.5 |
|---|---|---|---|
| **β = +2.5** | 무표정 남성 (수염) | 무표정 남성 | **미소 남성** |
| **β = 0**    | 무표정 청년 | baseline | 미소 청년 |
| **β = −2.5** | 무표정 여성 (금발) | 무표정 여성 | **미소 여성 (금발)** |

- α 모든 row에서 smile 일관 control
- β 모든 column에서 gender 일관 control
- 4 corner 모두 의도된 attribute 조합 정확히 실현

→ **1D sweep에서 부분적이었던 disentanglement가 2D plane에선 깔끔하게 분리**.
두 axis가 plane 위에서 동시에 표현 가능한 정도로 직교성 보장.

## Next steps
이게 본 연구의 가장 큰 visualization 성과. 이제 두 갈래:
1. 이 plane 위에서 **boundary 다시 그리기** — attribute region이 sign
   pattern으로 표현되는가? → **doc 09**
2. 채널 수준에서 **누가 smile, 누가 gender를 인코딩** 하는지 → **doc 09**
3. 그 정보로 **selective inject** — gender drift 없이 smile만 → **doc 10**
