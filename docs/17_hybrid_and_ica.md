# 17 — Subspace+supervision hybrid vs ICA rotation

Phase 25 + 26 · `scripts/31_subspace_supervision.py`, `scripts/32_ica_rotation.py` ·
figures `figures/exp26_subspace_supervision.png`, `figures/exp27_ica_rotation.png`

## 실험 목적
Phase 24의 12D h-Jacobian subspace는 entangled. 그 안에서 깨끗한 단일
concept axis를 뽑는 두 방법 비교:
- **Phase 25 (hybrid)**: supervised smile Δx를 subspace에 projection
- **Phase 26 (ICA)**: unsupervised ICA rotation으로 disentangled basis

## 왜 하는지
Phase 24 결론: 한 점에서 여러 방향 추출은 되나 singular basis가
concept-aligned 아님. 깨끗한 axis를 얻는 두 경로를 직접 비교.

## Phase 25 — Subspace + supervision hybrid → **성공**

### Capture ratio
supervised smile Δx (unit) 를 subspace에 projection한 norm:

| subspace | capture ratio |
|---|---:|
| Jacobian 12D | **0.333** |
| Random 12D (control) | 0.008 |

→ Jacobian subspace는 random보다 **40배** smile-aligned. 단 12D는 smile
방향의 33%만 포함 (67%는 밖).

### Decoded 비교 (`exp26_subspace_supervision.png`)
- **(a) raw supervised smile**: 명확한 smile 전이
- **(b) Jacobian subspace projection** (r=0.33): **(a)와 거의 동등**
- **(c) random subspace projection** (r=0.01): 변화 없음

**핵심 발견**: capture ratio 0.33인데도 projection을 unit으로
renormalize해 inject하면 raw supervised와 동등한 editing. → 잡힌 33%가
"의미적으로 effective한 성분"이고, 놓친 67%는 모델이 반응 안 하는 방향
(decode에 영향 없는 null-ish 성분). **하이브리드 작동**.

per-direction smile coefficient: v5 (0.18), v9 (0.18) 가 대부분 — Phase 24
의 cos(smile) 관찰과 일치.

## Phase 26 — ICA rotation → **약함**

12개 direction의 decoded ± effect에 FastICA(12) → 12×12 unmixing →
rotated basis.

| | original | ICA 후 |
|---|---:|---:|
| best \|cos(smile)\| | 0.179 | 0.204 |
| best \|cos(gender)\| | 0.180 | 0.231 |

→ **미미한 개선**. ICA는 통계적 독립성을 최적화하는데, 그게 "smile/gender
와 정렬"과 일치하지 않음. `exp27_ica_rotation.png`의 12개 ICA 방향도
여전히 entangled (각각 여러 attribute 변화).

ICA는 subspace가 담은 것 이상을 만들 수 없음 — subspace의 33% smile을
한 axis로 완전 집중시켜도 cos는 0.33이 한계인데, ICA는 0.20에 그침
(독립성 ≠ 정렬).

## 결론

| 방법 | 깨끗한 concept axis? |
|---|---|
| Subspace + supervision (Phase 25) | ✅ raw supervised와 동등한 editing |
| ICA rotation (Phase 26) | ❌ 미미한 개선, 여전히 entangled |

**소량의 supervision이 unsupervised rotation보다 훨씬 효과적.** Unsupervised
로 semantic subspace는 찾되 (h-Jacobian), 그 안의 깨끗한 axis를 뽑는
데는 label-derived direction의 projection이 정답. ICA처럼 순수
unsupervised rotation으로는 disentangle 안 됨.

→ 본 연구의 concept-direction 결론: **unsupervised subspace 발견 +
minimal supervision (axis 선택)** 하이브리드가 실용적 sweet spot.

## Next steps
- 더 큰 subspace (k=30, 50) → capture ratio가 1에 가까워지는가
- 다른 attribute (gender, age) 도 같은 hybrid로 검증
- Phase 25 (b) projection을 multi-step inject로 → editing 강도 정량 비교
