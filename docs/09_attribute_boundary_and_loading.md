# 09 — Boundary on attribute plane + per-channel attribute loading

Phase 14 + 15 · `scripts/15_attribute_boundary.py`, `scripts/16_channel_loading.py` ·
figures `figures/exp10_attribute_boundary.png`, `figures/exp11_channel_loading.png`

## 실험 목적
두 가지 병렬:
- **A (Phase 14)**: doc 08의 attribute plane 위에서 doc 01 식 sign-pattern
  boundary를 그림. attribute region이 boundary로 표현되는가?
- **B (Phase 15)**: 채널별로 attribute axis에 얼마나 loading 하는지 측정,
  분류.

## 왜 하는지
- (A) doc 01–02에서 random-anchor plane은 sign boundary가 의미를 못 가졌음.
  attribute-meaningful axis면 boundary가 attribute region에 직접 매핑될지
  궁금.
- (B) doc 08의 plane이 깔끔하게 disentangled하다는 건 model 내부에 두
  attribute가 어떻게든 분리되어 있다는 것. 채널 단위인가? 그렇다면
  selective inject (doc 10) 가능.

## 가설
(A) Sign boundary가 (smile, gender) attribute combination region 4개를 분할
(B) 채널은 (i) smile-pure, (ii) gender-pure, (iii) joint, (iv) weak로 분화

## 결과

### A: Sign boundary는 여전히 radial — attribute region 매핑 실패
`figures/exp10_attribute_boundary.png` (t=450, anchor 0 baseline + α·smile + β·gender):

- 396 active channel → 2432 region (saturation)
- top-K=20 → 159 region, 여전히 **(0, 0) 중심 radial** 패턴 (doc 02의 artifact 그대로)
- **단일 step sign pattern은 multi-step trajectory editing의 attribute
  효과를 직접 인코딩하지 않음**
- 정보가 있는 공간 (h)과 그 정보가 attribute로 표현되는 메커니즘(trajectory)
  이 분리되어 있다는 결론

→ **방법론적 finding**: diffusion h-space "decision boundary"는
classifier/GAN의 decision boundary와 의미가 다름. doc 04의 약한 신호와
일관.

### B: 채널 분화는 명확하게 정량화됨
`figures/exp11_channel_loading.png`:

`A_c = mean_spatial(Δh_smile_orth_c)`, `B_c = mean_spatial(Δh_gender_c)`

**Entanglement check**:
| | value |
|---|---:|
| corr(A_orig, B) | **+0.559** (강한 entanglement) |
| corr(A_orth, B) | **+0.134** (orth 후 거의 없음) ✓ |

**Channel categorization** (purity > 0.85):
| 분류 | 수 | 비율 | top 채널 예 |
|---:|---|---:|---|
| **smile-pure** | 110 | 21% | 503, 118, 119, 122 |
| **gender-pure** | 173 | 34% | 316, 412, 156, 79 |
| **joint** | 101 | 20% | 80 (multi-attr) |
| weak | 128 | 25% | — |

채널 80은 진정한 multi-attribute encoder.

→ **mid_block 512 채널이 attribute별로 분화 인코딩**. 이게 본 연구의
가장 actionable한 표현.

## Next steps
B가 actionable → doc 10:
- smile-pure 110 채널에만 inject → gender drift 제거되는가?
- multi-attribute 채널 (ch 80) 제외 시 효과?
- → **doc 10**
