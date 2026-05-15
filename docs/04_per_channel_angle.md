# 04 — Per-channel boundary + angle distribution

Phase 7 + 9 · `scripts/08_single_channel.py`, `scripts/10_channel_angle_stats.py` ·
figures `figures/exp3_single_channel.png`, `figures/exp5_channel_angle_stats.png`

## 실험 목적
**채널을 하나 고정**한 채 64개 spatial location의 boundary line을 plane에
그림. 각 channel의 내부 방향 편향 정량.

## 왜 하는지
- 같은 conv filter는 인접 pixel에 적용 → 인접 line은 비슷한 방향이어야 함
  (model의 구조적 smoothness)
- "어떤 channel이 한 방향성이 강한가" 가 의미 있다면 그 channel은
  특정 semantic axis에 align되어 있을 가능성 — concept direction 후보

## 가설
1. 같은 channel 내 64 line은 인접 spatial 위치 weight smoothness 때문에
   비슷한 기울기로 cluster될 것
2. 그 cluster 강도가 채널마다 다를 것 (어떤 channel은 강한 directional
   bias, 어떤 channel은 random)
3. 강한 bias의 채널은 "특정 attribute axis" 후보

## 결과

### Top-4 most-diverse channel (Phase 7)
386, 107, 131, 338. 64개 line이 plane을 분할. **방사형 없음** (doc 03과 일관).
채널 107, 131은 시각적으로 약한 directional cluster.

### Circular concentration R (Phase 9 정량)

각 channel의 64 line normal angle θ → doubled-angle representation `z = exp(2iθ)`:
- 평균 방향: μ = arg(⟨z⟩) / 2
- 집중도: R = |⟨z⟩| ∈ [0, 1]

```
R 분포 (512 channel):
  min    = 0.006
  median = 0.141
  max    = 0.425
```

**어느 채널도 R ≤ 0.43**. 가장 정렬된 채널조차 평행에 한참 못 미침.

Top-4 most-concentrated:
| ch | R | μ (deg) |
|---:|---:|---:|
| 357 | 0.425 | 3 |
| 111 | 0.425 | 15 |
| 362 | 0.371 | 8 |
| 291 | 0.355 | 5 |

→ Top 채널들은 평균 angle ~0–15° (horizontal normal). Bottom 채널들은
거의 균등 분포.

## Next steps
**중요한 부정 결과**: directional bias는 존재하지만 약함 (median R=0.14).
"단일 channel = 단일 concept axis" 가설은 약하게만 지지.

이게 doc 03 → doc 05의 pivot 신호:
- Channel-level geometry는 약한 신호 → 강한 supervised signal이 필요
- → **supervised Δh** (mean-shift over labeled images) 시도해야 함

→ doc 05
