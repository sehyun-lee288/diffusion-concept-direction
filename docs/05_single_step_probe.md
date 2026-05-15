# 05 — Single-step Δh probe (gate experiment) ✗

Phase 10 · `scripts/11_smile_probe.py` ·
figure `figures/exp6_smile_probe.png` + `figures/exp6_smile_probe_frames/`

## 실험 목적
h-space에 **supervised attribute direction이 있고 그 방향으로 inject하면
decoded image attribute가 변하는가** 라는 가장 직접적인 feasibility check.

(Independent review agent의 강한 권고: "이걸 안 하고 attribute-paired
anchor로 직진하면 null result에 시간 버림".)

## 왜 하는지
- doc 04: unsupervised channel geometry는 약한 신호만 줌
- doc 01의 sparse thumbnail: h-only inject가 visually weak — 원인 불명
- **이 실험이 통과하면 attribute axis 분석은 안전. 실패하면 skip-aware로 pivot.**

## 가설
1. `eurecom-ds/celeba-hq-256` (40 attribute label)에서 20 smile + 20
   no-smile sample. Mean-shift `Δh = mean(h|smile=1) − mean(h|smile=0)`.
2. `||Δh|| ≈ 80` (큰 magnitude), `||Δh|| / ||h_test|| ≈ 0.4`.
3. 따라서 s ∈ [−3, +3] sweep으로 inject하면 시각적 smile 변화 visible.

## 결과 — **FAIL**

### h-space에서는 Δh가 거의 perfect한 classifier
| 분류 | mean projection | d′ (separation) |
|---|---:|---:|
| smile=1 | +25.6 | |
| smile=0 | −53.0 | **4.23** |

→ h-space에 attribute 정보는 명확하게 인코딩되어 있음.

### 그런데 decoded image는 거의 변하지 않음
| 측정 | 값 |
|---|---:|
| `||source − decoded(s=0)||` (reconstruction noise floor) | **11.10** mean abs pixel |
| `||decoded(s=+3) − decoded(s=0)||` (injection signal) | 1.25 |
| **Signal / Noise ratio** | **0.11** (signal < noise floor) |

`s=+3` 은 projection이 −53 (no-smile 평균) → +183 (smile 평균 +25를 한참
초과) 인데도 decoded image는 거의 동일.

### 진단
**Encoder skip connection이 dominant**. U-Net은 x_500의 모든 spatial 정보를
encoder skip을 통해 직접 decoder로 전달. mid_block 한 곳만 바꿔도 출력
거의 안 변함. h-space는 attribute 정보를 *passive* 하게 담고 있을 뿐.

## Next steps
- (review agent 추천 1) **skip-aware injection** — Δskip + Δh
- (review agent 추천 2) **multi-step trajectory editing** — 매 step의
  inject가 누적되어 skip 자체를 bend
- 후자 (Asyrp 방식) 가 더 명확한 step이고, 위험도 낮음 → **doc 06**
