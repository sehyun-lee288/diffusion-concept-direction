# 10 — Selective-channel injection (clean editing in |s| ≤ 4)

Phase 16 · `scripts/17_selective_channel.py`, `scripts/18_smile_pure_boost.py` ·
figures `figures/exp12_selective_channel.png`, `figures/exp12b_smile_pure_boost.png`

## 실험 목적
doc 09의 채널 분류 (110 smile-pure, 173 gender-pure, ...) 를 inject scope에
적용. **smile-pure 채널에만 Δh inject** 해 gender drift 없는 editing 달성.

## 왜 하는지
- doc 11 (Phase 11) smile direction은 gender·age와 entangled
- doc 07 1D orthogonalization은 부분 fix만
- doc 08의 plane은 깨끗하지만 두 axis 동시 control이 필요
- doc 09: 채널이 attribute별로 분리되어 있다는 직접 증거
- → 가장 직접적인 fix: 채널 mask로 disallowed direction을 0 처리

## 가설
1. 110 smile-pure 채널만 활성 (다른 402 채널은 inject = 0) → gender stays fixed, smile changes
2. Leakiest 5 채널 (|A·B| 큰 122, 114, 372, 162, 118) 제외 시 더 깔끔
3. Smile signal magnitude는 줄겠지만 일정 s range까지는 disentangled

## 결과 — **3-variant 비교** + **boost test**

### Variant 비교 (figures/exp12_selective_channel.png, s ∈ [−3, +3])
| Row | Channel set | Gender drift | Smile 강도 |
|---|---|:---:|:---:|
| A (baseline) | all 512 | ❌ s=+3 → blonde 여성 | 강함 |
| B (smile-pure) | 110 | ✅ 동일 남성 유지 | 약함 |
| C (smile-pure − leak top-5) | 105 | ✅ 동일 남성 유지 | ≈ B |

→ **gender drift 완전히 제거됨**. 단 smile signal도 작아짐 (110/512 채널만
사용). B와 C 차이는 시각적으로 미미.

### Boost test (s ∈ [−6, +6], `figures/exp12b_smile_pure_boost.png`)
큰 s로 효과 증폭 가능한지:
- s ∈ [−4, +4]: **gender 안정, smile monotonic control** — clean editing range
- s = +6: 다시 **금발 여성으로 drift** — selective mask도 새어버림

→ **Clean editing range: |s| ≤ 4**. 그 이상에서는 downstream 비선형성
(LayerNorm, residual coupling) 이 attribute를 재결합.

### 해석
- Mid_block의 attribute structure는 rank-1 channel split로 잘 근사되지만 perfect 아님
- 110 smile-pure 채널은 절대 isolation이 아니라 "주로 smile 인코딩" 정도
- 실용적 disentangled editing: |s| ≤ 4 만족 → 시각적으로 강한 변화는 |s| = 3 정도이므로 사실상 사용 가능 범위

## Next steps
지금까지의 마지막 단계. 후속 thread:
- T1. **Soft-purity weighting** (binary mask 대신 `purity^k` 로 weight)
- T2. **Anti-injection**: Δsmile + (-Δgender) 동시 inject로 active drift cancel
- T3. **Time-stability**: t=450 categorization이 t=50, 950에서도 유지되는가?
- T4. **Skip-aware editing**: Δskip + Δh — review agent 추천 대안 path
- T5. **Multi-attribute joint orth** (smile, age, glasses, ...)

T3가 가장 cheap한 다음 step (cached Δh만으로 수행).
T4가 가장 fundamentally 중요 (encoder skip이 본질적 lock).
