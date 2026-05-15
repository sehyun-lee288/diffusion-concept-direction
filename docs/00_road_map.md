# Research Roadmap

ASCII mind-map of the experimental thread.
✓ = succeeded as hypothesized · ✗ = failed (informative) · ◇ = methodological finding · ★ = key pivot

```
                              ┌─────────────────────────────────────────────┐
                              │  RQ: diffusion h-space에 "decision boundary"     │
                              │  paradigm을 적용할 수 있는가? attribute 의미를       │
                              │  담은 region 구조가 나오는가?                       │
                              └────────────────────┬────────────────────────┘
                                                   │
                  ┌────────────────────────────────┴───────────────────────────────┐
                  │                                                                │
                  ▼                                                                ▼
     [A] Random-anchor plane (Phase 4-9)                       [B] Attribute-paired (Phase 10+)
        h_500을 3 임의 anchor로 span                                supervised Δh로 axis 정의
                  │                                                                │
       ┌──────────┼───────────┬─────────────┐                                       │
       ▼          ▼           ▼             ▼                                       ▼
    region      radial      per-pixel    per-channel                       ┌────────────────┐
    map         artifact ◇   boundary     boundary                          │ Single-step Δh │
    (doc 01)    (doc 02)     (doc 03)     (doc 04)                          │ injection ✗   │
                  │                                                          │ (doc 05)     │
                  │ "top-K-balanced 선택의                                       └──────┬─────────┘
                  │  self-fulfilling 결과"                                              │
                  ▼                                                                    ▼
                결론: single-step                                              ★ Multi-step
                sign pattern은 attribute                                      injection ✓
                structure를 직접 표현 못함                                       (doc 06)
                                                                                        │
                                                                       observation: smile↔gender
                                                                       entangled (corr=+0.56)
                                                                                        │
                                                                       ┌──────────┬─────┴─────┬────────────┐
                                                                       ▼          ▼           ▼            ▼
                                                                Orth Δh    Attribute    Channel       Selective
                                                                (doc 07)   plane        loading       injection
                                                                           (doc 08)     (doc 09)      (doc 10)
                                                                              │             │             │
                                                                              ▼             ▼             ▼
                                                                       2D plane이      512 채널이      smile-pure
                                                                       (smile, gender) attribute별   채널만 inject
                                                                       축으로 깔끔하게  분화         → gender drift
                                                                       disentangled    (110/173/    제거 |s|≤4
                                                                                       101/128)
                                                                                                          │
                                                                                                          ▼
                                                                                            한계: |s|>4에서 다시
                                                                                            entanglement 복원 →
                                                                                            soft purity, anti-inject,
                                                                                            skip-aware (future)
```

## Reading order

```
00 road map (this file)
   │
01 random-anchor sign-pattern boundary       ──→ baseline visualization
02 radial-pattern diagnostic                 ──→ first methodological finding
03 per-pixel single-spatial-location         ──→ "true" neuron boundary
04 per-channel + angle distribution          ──→ within-channel direction bias is weak
05 supervised Δh single-step probe ✗         ──→ encoder skip dominance proven
06 multi-step (Asyrp-style) injection ✓      ──→ KEY pivot: skip lock broken
07 per-timestep orthogonalization            ──→ smile-gender corr 0.56→0.13
08 attribute-paired 2D plane                 ──→ disentangled semantic space
09 attribute-plane boundary + channel load   ──→ 110 smile-pure / 173 gender-pure
10 selective-channel injection               ──→ clean editing |s| ≤ 4
```

## Threads to pursue next (queued)

```
T1. soft-purity weighting (vs binary mask)              — refine channel selection
T2. anti-injection (Δsmile + (-Δgender))                — active drift compensation
T3. classification stability across t                    — does the same channel mean smile at t=50 and t=950?
T4. skip-aware editing (Δskip + Δh)                      — finish the unfinished pivot
T5. multi-attribute joint orth (smile, age, glasses…)   — beyond pair-wise
T6. trajectory-aware boundary                            — sign patterns across all t, not just t=500
```

## File map

| Doc | Phase(s) | Figure(s) |
|---|---|---|
| 01 | 4-5 | exp1_boundary.png |
| 02 | 5.5 | exp1_radial_analysis.png |
| 03 | 6 | exp2_single_location_pix4_4.png |
| 04 | 7, 9 | exp3_single_channel.png, exp5_channel_angle_stats.png |
| 05 | 10 | exp6_smile_probe.png |
| 06 | 11 | exp7_multistep_smile.png |
| 07 | 12 | exp8_orthogonalization.png |
| 08 | 13 | exp9_attribute_plane.png |
| 09 | 14, 15 | exp10_attribute_boundary.png, exp11_channel_loading.png |
| 10 | 16 | exp12_selective_channel.png, exp12b_smile_pure_boost.png |
| 11 | 17, 17b, 17c, 17d | exp15_egbas_analog, exp16_egbas_multistep, **exp17_selective_polytope**, **exp18_trajectory_polytope** |
| 01 b | 4-5 (bonus) | **exp13_random_anchor_all_lines.png** (전체 472 line) |
| 09 b | 13-15 (bonus) | **exp14_attribute_plane_class_lines.png** (category 색칠 + thumbnails) |

Related work survey: see `related_work/01-05_*.md`.
Full execution log: see `IMPLEMENTATION_PLAN.md`.
Analytical write-up: see `FINDINGS.md`.
