# 13 — Bifurcation test + speciation-point editing

Phase 20 + 21 · `scripts/26_bifurcation_test.py`, `scripts/27_speciation_editing.py` ·
figures `figures/exp21_bifurcation.png`, `figures/exp22_speciation_editing.png`

## 실험 목적
Phase 18-19가 찾은 speciation zone (t≈700) 을 두 각도로 검증:
- **Phase 20**: t≈700이 진짜 *bifurcation* (sharp basin-flip) 인가, 아니면
  smooth 전이인가
- **Phase 21**: editing을 speciation window에 집중하면 더 효율적인가

## 왜 하는지
- "Polytope 구할 수 있나" 의 최종 답을 내려면 boundary가 hard (sharp)
  인지 soft (continuous) 인지 알아야 함
- 우리는 줄곧 t=500에서 editing — Phase 18은 t≈700이 commit zone이라
  했으므로 editing 위치를 검증해야 함

## 가설
- Phase 20: symmetry-breaking 이론대로면 t≈700에서 작은 perturbation이
  basin을 flip → smile_score(s) 가 sharp step
- Phase 21: speciation window editing이 all-step editing과 동등하거나
  더 효율적

## 결과

### Phase 20 — Bifurcation test (`exp21_bifurcation.png`)

x_700을 smile 방향 Δx로 perturbation, smile_score(s):

| s | −6 | −3 | 0 | +3 | +6 |
|---|---:|---:|---:|---:|---:|
| score (Δx dir) | −289 | −236 | −89 | +96 | +236 |
| score (random) | −90 | −89 | −89 | −89 | −89 |

- **Smooth sigmoid ramp — sharp step 아님**. 중앙 (s∈[−1,+1]) slope ≈ 62/unit,
  edge slope ≈ 14–41/unit — 약한 central steepening은 있으나 discontinuity
  전혀 없음
- Random 방향 control은 flat (~−89) → Δx 방향 자체는 의미 있음
- Decoded strip: 무표정 소녀 → 중립 → 활짝 웃음, 연속 전환

→ **t≈700에 sharp bifurcation 없음**. Speciation은 attribute가 *encode*
되는 지점이지 basin-flip knife-edge가 아님. (Biroli et al.의 speciation은
무한 데이터 극한에서 sharp; 유한 실제 모델에서는 smooth — 우리 결과가
바로 그 finite-size 거동.)

### Phase 21 — Speciation-window editing (`exp22_speciation_editing.png`)

Δh injection을 timestep window별로 제한:

| Window | injected steps | editing 효과 |
|---|---:|---|
| A) all | 50 | 강함 (s=−3 소녀 → s=+3 큰 웃음) |
| **B) speciation [600,800]** | **11** | **A와 동등** |
| C) early [350,550] | 10 | 약함 (거의 변화 없음) |

- **B (speciation window, 11 step) 이 A (all, 50 step) 와 동등한 editing**
  — 1/5 injection으로 같은 효과
- **C (off-speciation, 10 step) 는 step 수가 B와 비슷한데도 훨씬 약함**
- → editing leverage는 step 수가 아니라 *위치*에 있음. speciation zone이
  editing이 "먹히는" 구간

**중요**: 우리가 줄곧 쓴 t=500은 window C 범위 (350–550). 약한 zone이었음.
진짜 leverage는 t≈700.

## "Polytope 구할 수 있나" — 최종 답

| | 존재? | 형태 |
|---|:---:|---|
| Hard polytope (닫힌 다면체) | ❌ | Phase 20이 sharp boundary 부재 확인 |
| Soft speciation region | ✅ | t≈700 부근 연속 전이 zone, 경계가 smooth sigmoid |

- E-GBAS식 hyperplane 교집합 polytope은 (a) 계산은 되지만 (Phase 4-17)
  의미 없고 (b) 진짜 boundary는 sharp하지도 않음
- Diffusion attribute "boundary" = **soft speciation region**.
  Hyperplane이 아니라 smooth sigmoid 전이. 닫힌 수식 없음
- 단, 그 region의 **editing leverage 위치** (t≈700) 는 명확히 특정 가능
  (Phase 21)

## Next steps
- speciation window를 더 좁혀 (예: t∈[680,720]) 최소 effective window 탐색
- gender / age 의 speciation window — attribute마다 다른 위치인가
- Phase 20을 다른 timestep (t=500, 900) 에서 반복 → speciation 전후로
  ramp 기울기가 어떻게 변하는가 (t=900이면 더 flat할 것으로 예측)
