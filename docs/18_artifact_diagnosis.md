# 18 — Artifact diagnosis for h-Jacobian direction sweeps

Phase 27 · `scripts/33_artifact_diagnosis.py` ·
figure `figures/exp28_artifact_diagnosis.png`

## 실험 목적
exp25 (Phase 24) 의 12개 h-Jacobian direction sweep에서 어떤 decode는
깨끗한 얼굴, 어떤 건 artifact. **무엇이 artifact를 만드는가** 규명.

## 왜 하는지
사용자 관찰: figure에서 artifact 행과 얼굴 행이 섞여 있음. 원인을 알면
clean edit을 위한 가이드라인을 얻을 수 있음.

## 가설 (3개)
- **H1 σ imbalance**: 높은 σ 방향이 h를 더 세게 밀어 off-distribution
- **H2 \|s\| onset**: perturbation 크기가 threshold 넘으면 깨짐
- **H3 manifold curvature**: 한 sign은 manifold 벗어남, 반대는 따라감
  (sign 비대칭)

## 방법
Artifact score = real-face PCA basis (60 real CelebA face, 64×64,
PCA-40) 에 대한 **relative residual**. 얼굴이면 residual 낮음, artifact면
높음. 각 12 방향 × signed s-sweep {−66 … +66} decode.

Calibration: 실제 얼굴 residual ≈ 0.20 (in-sample), 순수 noise ≈ 0.67.
s=0 base decode (깨끗한 generated face) ≈ 0.55.

## 결과

| 가설 | 판정 | 근거 |
|---|---|---|
| **H2 \|s\| 크기** | ✅ **주원인** | 12 방향 전부 V자 — s=0에서 0.55, \|s\| 커질수록 residual 단조 증가 |
| **H1 σ rank** | ✅ 부차 (역방향) | corr(σ, max-artifact) = **−0.52** — 높은 σ가 *덜* artifact |
| **H3 sign 비대칭** | ⚪ 미미 | mean asymmetry 0.056, 대체로 대칭 |

### 상세

**주원인 — perturbation 크기**: 모든 direction이 s=0 (residual 0.55,
깨끗한 얼굴) → \|s\|=44 (0.69~0.84) 로 단조 상승. Phase 24가 쓴
s≈44는 이미 모든 방향을 face manifold에서 일부 벗어나게 하는 크기.

**σ-rank modulation**: 같은 \|s\|라도
- top-σ (v1~v4, σ=28~42): residual 낮음 (~0.68~0.73) → 더 얼굴같음
- tail-σ (v9, v10, v12, σ≈20~22): residual 높음 (~0.80~0.84) → artifact

→ exp25의 "얼굴 행" = top-σ 방향, "artifact 행" = tail-σ 방향.

## 해석

**왜 high-σ가 더 깨끗한가**: h-Jacobian의 top singular vector는 모델이
가장 자연스럽게 얼굴을 변형시키는 축 — 그 방향으로 밀면 학습된 face
manifold를 따라감. Tail singular vector (낮은 σ) 는 덜 구조적 →
조금만 밀어도 manifold를 벗어나 artifact.

이건 diffusion geometry 문헌의 manifold-tangent vs manifold-normal
구분과 일치: top-σ 방향은 manifold-tangent에 가깝고, tail-σ 방향은
manifold-normal 성분이 큼.

## 실용적 함의

깨끗한 (non-artifact) h-Jacobian edit:
- **\|s\| ≤ 22** 정도로 작게 유지
- **top-σ 방향 우선** — tail 방향은 artifact-prone
- 또는 per-direction으로 σ에 반비례하게 s 조절

## Next steps
- s를 작게 (\|s\|≤22) 한 깨끗한 12-direction figure 재생성
- top-σ vs tail-σ 방향의 manifold-tangent/normal 분해 정량
  (Stanczuk et al. 방식의 score-normal 추정)
