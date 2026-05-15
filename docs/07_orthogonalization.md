# 07 — Per-timestep orthogonalization (smile ⊥ gender)

Phase 12 · `scripts/13_orthogonalization.py` ·
figure `figures/exp8_orthogonalization.png` + `figures/exp8_orthogonalization_frames/`

## 실험 목적
Doc 06의 smile-gender entanglement를 InterFaceGAN 스타일 Gram-Schmidt로
**post-hoc 제거**. Per-timestep 단위로 적용.

## 왜 하는지
- doc 06: `Δh_smile`이 gender 차이를 함께 인코딩 → identity drift
- mean-shift는 표본의 모든 covariate skew를 끌고 옴
- 가장 simple한 fix: gender direction을 supervised로 뽑고 smile에서 빼기

## 가설
1. Δh_gender도 동일 방법으로 추출
2. Per-t: `Δh_smile^⊥_t = Δh_smile_t − ⟨Δh_smile_t, Δh_gender_t⟩ / ‖Δh_gender_t‖² · Δh_gender_t`
3. orth direction으로 sweep 시 identity (gender) 안정, smile만 변할 것

## 결과 — **부분 success**

### Gender component magnitude (전체 norm의 44%)
| t | ‖smile‖ | ‖smile⊥‖ | gender comp |
|---:|---:|---:|---:|
| 50 | 84.1 | 75.5 | **37.1** |
| 450 | 79.9 | 71.7 | 35.2 |
| 750 | 54.5 | 49.9 | 22.0 |

→ smile direction norm의 약 **44%가 gender 성분**. Sample size 작아서가
아닌 substantial confound.

### Side-by-side sweep (figures/exp8_orthogonalization.png)
- Row 1 (original): s -3 어린 여자 → s +3 미소 남성. 큰 identity drift.
- Row 2 (orth): s ∈ [−3, +2]는 동일 남성, s = +3에서 다시 gender drift.

→ **부분 decoupling**. 1차원 gender axis만 빼는 걸로는 부족. Multiple
confounder (age, hair, ...) 까지 빼야 더 깨끗.

## Next steps
- 2D plane 자체로 표현하면 두 axis가 동시에 표현 가능해 더 깨끗할 가능성
  → **doc 08**
- 더 많은 confounder 동시 orth (T5, future)
- Soft-purity weighting (T1, future)
