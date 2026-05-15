# Toy Experiment 1 — 분석 결과

DDPM CelebA-HQ의 U-Net bottleneck (h-space)에서 3개 anchor가 span하는 2D
plane 위 boundary 구조 분석. 각 발견은 figure + 진단 script로 뒷받침된다.

- 실험 코드: `scripts/`, `diffusion_boundary/`
- Phase 진행: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- 연구 동기/RQ: [research_plan.md](research_plan.md)

---

## 0. 셋업 요약

| 항목 | 값 |
|---|---|
| Model | `google/ddpm-celebahq-256` |
| Timestep | t = 500 |
| h | `unet.mid_block` 출력, shape `(1, 512, 8, 8)` |
| Plane basis | non-unit orthogonal; anchors at (0,0), (1,0), (a, 1) |
| Grid | 50×50, (α, β) ∈ [−0.5, 1.5]² |
| Anchors | CelebA-HQ index 0 / 1000 / 5000 (`mattymchen/celeba-hq`) |
| Inversion | forward diffusion (q-sample), fixed noise seed per anchor |

각 channel `c`의 boundary는 plane 위에서 **선형**:
> `A_c · α + B_c · β + C_c = 0`,
> `A_c = mean_c(h2 − h1)`, `B_c = mean_c(v)`, `C_c = mean_c(h1)`
> (또는 `mean` 대신 `[i*, j*]` 단일 pixel — 정의에 따라 다름)

3개 점을 지나는 plane은 affine 의미에서 정확히 유일.

---

## 1. Sign reduction 선택이 결정적

3개 anchor의 channel-wise sign vector 간 Hamming distance를 reduction별로 측정:

| reduction | (h1, h2) | (h1, h3) | (h2, h3) | 의미 있나? |
|---|---:|---:|---:|---|
| **spatial max** | 0/512 | 0/512 | 0/512 | ❌ 모든 channel이 항상 +1 |
| **spatial mean** | 215/512 | 215/512 | 228/512 | ✅ |
| **spatial median** | 212/512 | 214/512 | 228/512 | ✅ |

→ DDPM CelebA-HQ의 mid_block 출력은 SiLU residual을 여러 번 거친 결과로
모든 channel이 적어도 한 spatial cell에서 양수. 따라서 spatial-max는
정보를 보존하지 않는다. **spatial-mean** 또는 **median**이 채택 가능.

본 실험은 spatial-mean 채택 → Phase 4 sign grid 생성.

---

## 2. Plane basis는 unit-norm이 아니어야 한다

처음 build_plane을 orthonormal로 구현하니 anchor 1이 α ≈ 278.7 (= ‖h₂−h₁‖)
에 위치 → 임의의 `α ∈ [−0.5, 1.5]` grid가 anchor 영역을 거의 포함 못 함 →
50×50 grid가 단 3개 unique sign pattern만 가짐.

수정 후 (orthogonal but not unit-norm; anchors at (0,0), (1,0), (a,1)) →
2432 / 2500 unique pattern. **anchor가 grid의 round 좌표에 오도록 basis를
calibrate**하는 게 모든 후속 분석의 전제.

---

## 3. Top-20 region map은 "방사형"으로 보이지만 artifact다

`figures/exp1_boundary.png` (top-K = 20 most-balanced channels) 에서
boundary line들이 (α, β) ≈ (0.5, 0.5) 근처로 수렴하는 듯 보임.

### 3.1 이론

Plane 위에서 line이 한 점을 공유하려면 `M = [A | B | C] ∈ ℝ^{512×3}`이
rank ≤ 2여야 함 (right null space ≠ {0}).

### 3.2 측정

SVD of `M`: singular values **[11.17, 8.24, 3.85]**, energy share
**60.2 / 32.7 / 7.1 %**. 효과적인 rank ≈ 2지만 정확히 2는 아님 — s3가
7%를 차지. SVD 기반 best common point는 (−0.785, −0.323)으로,
실제 cluster 위치 (0.495, 0.500)과 일치하지 않는다.

### 3.3 K-sweep — 결정적 증거

| K | n_pairs | median (α, β) | MAD |
|---:|---:|---:|---:|
| 10  | 45      | (+0.500, +0.502) | (0.007, 0.005) |
| 20  | 190     | (+0.495, +0.500) | (0.012, 0.013) |
| 50  | 1 225   | (+0.504, +0.498) | (0.036, 0.035) |
| 100 | 4 950   | (+0.488, +0.478) | (0.073, 0.061) |
| 250 | 31 125  | (+0.534, +0.433) | (0.184, 0.203) |
| 512 | 130 816 | (+0.647, +0.274) | (0.456, 0.474) |

K가 커질수록 cluster가 흩어진다. K=512 (전체)의 MAD 0.46은 거의 grid 폭
수준으로 dispersed.

### 3.4 인과 설명

Top-K-balanced 기준 = "grid window 내에서 ±1 비율이 50:50에 가까운 channel" =
boundary line이 **grid 중앙을 지나는 channel**. 따라서 그 선택된 channel
들의 line은 정의상 grid 중앙 (0.5, 0.5) 부근에 모인다. **자기충족적
artifact**.

추가로 모델 자체의 rank-2-ish 구조 (s₁/s₂가 92.9% energy)가 이 artifact를
시각적으로 더 깔끔하게 만들어준다.

진단: `figures/exp1_radial_analysis.png`,
script: `scripts/06_analyze_radial.py`.

---

## 4. 단일 spatial location에서 본 진짜 neuron boundary는 uniform

방사형이 artifact인지 확인하려면 reduction 없이 **개별 neuron** boundary를
보면 됨. Pixel `(i*, j*)` 고정, `sign(h_c[i*, j*])` for all c.

`scripts/07_single_location.py` 실행 (center pixel (4, 4)):
- 475 / 512 channels active (window에서 sign이 변함)
- 2438 regions / 2500 cells — grid 해상도 한계
- **`figures/exp2_single_location_pix4_4.png` 오른쪽 panel — 475 line이
  방향·offset 모두 ~uniform**

→ Phase 3의 가설 확정. 방사형은 모델 특성이 아니라 선택 편향.

---

## 5. 채널 내부에는 방향 편향이 **약하지만** 존재한다 (정량)

채널 `c` 고정, 64개 spatial location `(i, j)`의 line을 plane에 그리면 어떤가?
`scripts/08_single_channel.py` 가 top-4 most-diverse channel (386, 107, 131,
338)을 보여주고, `scripts/10_channel_angle_stats.py` 가 모든 512 채널의
정량 분포를 제공.

### 5.1 정의

각 line의 normal vector `(A_c[i,j], B_c[i,j])`의 angle을 `θ ∈ [0, π)`로
보면 (line direction은 ±방향 동일) circular stats는 doubled-angle
representation `z = exp(2iθ)` 위에서 정의:

- **평균 방향**: μ_c = arg(⟨z⟩) / 2
- **집중도**: R_c = |⟨z⟩| ∈ [0, 1]
  - R → 1: 64개 line 모두 평행
  - R → 0: 방향 균등 분포

### 5.2 측정 결과 (`figures/exp5_channel_angle_stats.png`)

```
R distribution across 512 channels:
  min = 0.006,  median = 0.141,  max = 0.425
```

**채널 어느 것도 강한 directional bias를 가지지 않음 (R ≤ 0.43)**. 즉
Phase 7의 "channel 107/131 line이 비슷한 기울기" 관찰은 정량적으로는
약한 효과.

| 분류 | channel | R | μ (deg) |
|---|---:|---:|---:|
| 가장 집중 | 357 | 0.425 | 3.0 |
| ↓        | 111 | 0.425 | 15.1 |
| ↓        | 362 | 0.371 | 8.2 |
| ↓        | 291 | 0.355 | 4.8 |
| 가장 분산 | 486 | 0.006 | 112.9 |
| ↑        | 224 | 0.006 | 47.4 |
| ↑        | 313 | 0.010 | 57.5 |
| ↑        | 470 | 0.015 | 118.8 |

### 5.3 관찰

- **Top-4의 μ가 0–15°에 몰림** (horizontal normal → vertical line).
  그 정도로 약한 효과지만 일관된 방향성이 존재.
- **Bottom-4는 거의 random**. Rose plot이 균등 ring 형태.
- 원인: 같은 channel = 같은 conv filter → 인접 `(i, j)` 계수가 smooth →
  인접 line이 비슷하게 tilt. 하지만 64개 모두 통계적으로 정렬되기엔
  부족.

### 5.4 함의

요약:
- **채널 간**: 방향 분포 ~uniform (Section 4)
- **채널 내**: 평균 R ≈ 0.14, 최대 0.43 — 약한 directional bias

따라서 "channel = semantic concept direction" 가설은 약하게만 지지됨.
강한 attribute control을 위해 채널을 선택하더라도, 그 채널의 64 line이
한 방향으로 align되어 있지는 않으므로 단일 axis 해석은 신중해야 함.

---

## 6b. (Phase 10 confirmation) Supervised Δh probe — h-only injection 실패

Phase 5의 "h-only injection은 약하다"를 supervised direction으로 stress-
test. Result: **decisive failure**. h-space에 attribute 정보는 명확히
있지만, decoder가 그것을 무시.

### Setup (`scripts/11_smile_probe.py`)

- `eurecom-ds/celeba-hq-256`에서 N=20 smile=1 image, N=20 smile=0 image
- 각 이미지 → q-sample (t=500) → mid_block h 캡처
- `Δh = mean(h | smile=1) − mean(h | smile=0)`, **||Δh|| = 78.6**
- Test image: 첫 no-smile (frontal). `||h_test|| = 194.1`,
  ratio `||Δh|| / ||h_test|| = 0.41` (큰 perturbation)
- 7-point sweep: s ∈ {−3, −2, −1, 0, +1, +2, +3} → inject `h_test + s·Δh`

### Δh는 h-space에서 smile을 정확히 분리

각 h-vector을 Δh에 projection:

| 분류 | mean | std | d′ (separation) |
|---|---:|---:|---:|
| smile=1 | +25.6 | 15.6 | |
| smile=0 | −53.0 | 21.6 | **4.23** |

d′ ≈ 4.2는 거의 perfect classifier 수준 (∼99% accuracy 예상).
**h-space는 smile 정보를 클리어하게 인코딩**.

### 그런데 decoded image는 거의 변하지 않음

| Diff metric | value |
|---|---:|
| 노이즈 floor: `\|source − decoded(s=0)\|` mean abs | **11.10** |
| s=+3 injection 효과: `\|decoded(s=+3) − decoded(s=0)\|` | 1.25 |
| signal-to-noise ratio | ≈ 0.11 (signal **below** floor) |

s=+3에서 projection은 −53 → +183 (smile class mean인 +25를 한참 초과)
인데도 decoded image는 baseline에서 ~10%만 변함. 노이즈 floor 아래.

Figure: `figures/exp6_smile_probe.png`, 개별 frame:
`figures/exp6_smile_probe_frames/`.

### 진단

원인은 **encoder skip connection의 dominance**. U-Net의 skip은 x_500의
모든 spatial 정보를 직접 decoder로 전달하므로, mid_block의 bottleneck
하나만 바꿔도 출력은 거의 변하지 않음. h-space는 smile 정보를 *passive*
하게 담고 있을 뿐 (linear probe로 잘 읽을 수 있음), 인과적 신호가 약함.

### 결론 및 pivot

- **단일 step (t=500 only) injection은 dead.** Supervised direction
  도 시각적 변화 없음 → unsupervised plane 분석은 더 약할 수밖에.
- 두 가지 가능한 pivot: skip-aware single-step, 혹은 multi-step
  trajectory editing (Asyrp 식). 후자를 먼저 시도 (§6c).

---

## 6c. Multi-step h trajectory editing — **SUCCESS**

Phase 10 실패의 진단(encoder skip이 dominant)을 **모든 denoising step에서
inject**해 우회. 매 step 약간씩 bend된 x_t가 다음 step의 encoder input이
되므로 skip 자체가 누적적으로 modified — single-step에서 안 풀린 잠금이
trajectory level에서 풀림.

### Setup (`scripts/12_multistep_smile.py`)

- 같은 20+20 dataset에서 t ∈ {50, 150, 250, …, 950} 각각의 Δh_t 계산
  (10개 timestep)
- ||Δh_t||: t=50에서 84 → t=950에서 12 (저 t에서 attribute signal 강함)
- 50-step DDIM denoising, **매 step의 mid_block 출력에 `+ s·Δh_{t_nearest}`
  를 hook으로 추가**
- Fixed x_T (seed=0), s ∈ {-3, …, +3} sweep

### 결과 (`figures/exp7_multistep_smile.png`)

s 축을 따라 **명확한 smile transition** 관찰:

| s | 관찰 |
|---:|---|
| −3 | 무표정, 어린 여자아이 |
| −1 | 무표정, 어린 남자아이 |
|  0 | 무표정, 성인 남성 (baseline) |
| +1 | 약한 미소, 중년 남성 |
| +3 | **명확한 미소**, 중년 남성 |

→ **multi-step injection은 동작**. Skip dominance는 단일 step inject의 문제였지
trajectory 자체의 한계가 아니었음.

### 단, identity 동시 변화 (entanglement)

s가 변할 때 smile뿐 아니라 **성별·나이·identity도 함께 변함**. 원인:

- N=20 per class 의 sample size에서 covariate balance가 불완전
- Our smile=1 sample은 우연히 더 남성·노년 비율이 높음
- Δh_smile = mean(smile=1) − mean(smile=0) 는 모든 covariate 차이의 합

해결책 (Phase 12 후보):
1. **Orthogonalization**: Δh_smile에서 Δh_gender, Δh_age 성분을 빼내
   pure smile axis 구성 (InterFaceGAN/StyleGAN concept axis 방식)
2. **Larger N + balanced sampling**: 각 covariate를 controlled로 한 sample
3. **Logistic probe + counterfactual regularization**

---

## 6. Decoding 결과 — h-only injection의 effect는 미묘

`figures/exp1_boundary.png`의 9개 thumbnail은 anchor 0의 `x_500`을 fixed
encoder input으로 두고 mid_block 출력만 grid point `h(α, β)`로 교체한 후
DDIM 20-step decode한 결과.

- 9개 모두 비슷한 정체성을 유지, 표정·헤어라이팅 정도만 미세 변화
- 이유: U-Net의 encoder skip connection이 모든 spatial 정보를 그대로
  decoder로 전달 → mid_block 단독 교체는 high-level 일부에만 영향
- Asyrp가 보고한 latent-diffusion 환경의 풍부한 semantic editing 효과보다
  훨씬 약함

⇒ **DDPM CelebA-HQ의 h-space는 단독으로 semantic-rich latent로 쓰기 어렵다**.
Latent diffusion이 가지는 컴팩트한 latent와는 성질이 다름.

---

## 6d. Orthogonalization과 Attribute-paired plane (Phase 12 + 13)

### 6d.1 Orthogonalization (Phase 12)

Per-timestep Gram–Schmidt로 smile에서 gender 성분 제거:

`Δh_smile^⊥_t = Δh_smile_t − ⟨Δh_smile_t, Δh_gender_t⟩ / ‖Δh_gender_t‖² · Δh_gender_t`

Norm 측정:

| t | ‖smile‖ | ‖smile⊥‖ | gender component |
|---:|---:|---:|---:|
|  50 |  84.1 |  75.5 | 37.1 |
| 450 |  79.9 |  71.7 | 35.2 |
| 750 |  54.5 |  49.9 | 22.0 |
| 950 |  11.7 |  10.6 |  5.0 |

→ gender 성분이 smile direction norm의 **44%**. 사소한 entanglement가
아니라 substantial한 covariate confound. 작은 N=20에서 sample skew가
주된 원인.

`figures/exp8_orthogonalization.png` (2-row sweep, same x_T):

- Row 1 (original smile): identity가 s 따라 크게 drift
  (어린 여자 → 무표정 남자 → 미소 남자)
- Row 2 (orth smile): identity 더 stable (-3 ~ +2까지 일관된 남자),
  하지만 s=+3에서 다시 gender drift

→ **부분적 success**. 1차원 gender axis만 빼는 걸로는 부족 — multiple
confounder (age, skin, hair) 까지 빼면 더 좋을 듯.

### 6d.2 Attribute-paired plane (Phase 13)

`α·Δh_smile_orth + β·Δh_gender` 를 매 step inject, 5×5 grid 디코딩.
`figures/exp9_attribute_plane.png` 관찰:

```
β = +2.5  [neutral male, mustache]  →  →  [smile male]
β = +1.25 [neutral male]            →  →  [smile male]
β = 0     [neutral young male]      →  →  [smile young]
β = −1.25 [neutral female]          →  →  [smile female]
β = −2.5  [neutral female blonde]   →  →  [smile female blonde]
  α=−2.5    α=−1.25    α=0    α=+1.25   α=+2.5
```

- α 축 = smile (왼→오 무표정 → 미소) — 모든 row에서 일관
- β 축 = gender (아래→위 여성 → 남성) — 모든 column에서 일관
- **4 corner 모두 의도된 attribute 조합 정확히 생성**

→ **plane 자체가 disentangled semantic space로 동작**. 1D orthogonalization
sweep에서 부분적이었던 분리가, 2D plane에서는 axes가 동시에 표현 가능해
훨씬 깨끗.

### Phase 13의 함의

원래 본 연구의 "boundary on plane" 패러다임이 attribute-paired axis로
의미를 회복함:

- 이전: 3개 random anchor가 span하는 plane → boundary는 artifact 위주
- 지금: `(Δh_smile, Δh_gender)`가 span하는 plane → 각 attribute가 한 축에
  매핑되어 sign-pattern boundary가 직접 attribute region을 인코딩할
  여지가 있음

다음 후보 (Phase 14): 이 attribute-paired plane 위에서 sign-pattern
boundary 다시 그리고, region이 attribute combination에 대응하는지 검증.

---

## 6e. Attribute-plane boundary + per-channel loading (Phase 14 + 15)

attribute-paired plane이 의미 있는 visualization을 만들었으니, 이제
원래 Phase 4 식 sign-pattern boundary 분석을 attribute axis 위에서
다시 시도. 동시에 per-channel attribute loading 분석으로 채널 분화
구조를 정량.

### 6e.1 Sign boundary on attribute plane (Phase 14)

t=450 (cache에서 t=500과 가장 가까운 키) 의 직접 계산:

`h(α, β) = h_anchor0 + α · Δh_smile_orth + β · Δh_gender`
→ 채널별 `sign(spatial_mean(h_c))`

`figures/exp10_attribute_boundary.png`:
- 396 active channel → 2432 region (full saturation; grid 해상도 한계)
- top-K=20 → 159 region, **여전히 radial 패턴** (이번엔 (0, 0) 중심)
- top-K-balanced criterion이 attribute axis에서도 in-window 중앙
  passing line을 선호 → Phase 5.5의 artifact가 재현됨

→ **단일 step sign pattern은 multi-step trajectory editing의 attribute
effect를 직접 인코딩하지 못함**. 디코딩된 이미지(exp9)는 깔끔한
attribute region을 보이지만 t=450의 sign pattern은 random radial. 정보가
single-step h에 있지만 그것이 decoded image의 attribute로 변환되는
경로는 trajectory level에서 일어남.

이건 본 연구의 큰 메시지: **diffusion model 'h-space boundary'는
classifier/GAN의 decision boundary와 의미가 다름**. h-space에 정보는
있지만 (probe d′=4.23), boundary geometry는 attribute structure를
fully 표현하지 못함.

### 6e.2 Per-channel attribute loading (Phase 15)

각 채널 c의 attribute loading: `A_c = mean_spatial(Δh_smile_orth_c)`,
`B_c = mean_spatial(Δh_gender_c)`.

`figures/exp11_channel_loading.png` 핵심 수치:

| 지표 | 값 | 해석 |
|---|---:|---|
| corr(A_orig, B) | **+0.559** | original smile direction이 gender와 강한 양의 상관 — entanglement |
| corr(A_orth, B) | **+0.134** | orthogonalization 후 상관 거의 사라짐 ✓ |
| median \|A_orth\| | 0.083 | 약하지만 살아있는 분포 |
| median \|B\| | 0.100 | |

채널 분류 (`purity = max(|A|, |B|) / hypot(A, B) > 0.85`):

| 분류 | 수 | top 채널 예 |
|---:|---|---|
| **smile-pure** | 110 | 583, 118, 119, 80 |
| **gender-pure** | 173 | 316, 412, 156, 79 |
| **joint** | 101 | 462, 405, 412, 81 |
| weak | 128 | — |

→ **512개 mid_block 채널이 attribute별로 분화 인코딩**. Gender 채널이
가장 많고 (173/512 ≈ 34%), smile은 그 절반 정도. 약 20%는 두 attribute
모두 인코딩하는 multi-attribute neuron.

채널 80은 smile-top과 gender-top 양쪽에 등장 — 진정한 multi-attribute
encoder의 예.

### 함의

1. **Boundary geometry 한계**: top-K-balanced + single-step sign 방식은
   attribute-meaningful axes에서도 radial artifact를 재생산. Boundary
   visualization 자체는 attribute structure를 직접 보여주지 않음.
2. **Channel-level 정보는 풍부**: 각 채널이 어느 attribute에 loading
   하는지가 정량적으로 분리 가능. 이게 본 연구가 도달한 가장 활용성
   높은 표현.
3. **Selective injection 가능성**: smile-pure 채널만 inject (gender-pure
   는 그대로) 하면 더 disentangled editing이 가능할 수 있음 (Phase 16
   후보).

## 6f. Selective-channel injection (Phase 16)

Phase 15의 채널 분류를 사용해 inject scope를 좁힘. 가설: smile-pure
110 channel에만 Δh inject하면 gender drift 없이 smile만 변할 것.

### Setup (`scripts/17_selective_channel.py`)

세 variant 비교, 동일 x_T (seed=0), s ∈ {−3, …, +3}:

- **A** 모든 512 channel inject (Phase 11 reproduction)
- **B** smile-pure 110 channel만 (다른 402 channel은 0)
- **C** smile-pure 105 channel (B에서 |A·B| 상위 5개 leaky 제거: 122,
  114, 372, 162, 118)

### 결과 (`figures/exp12_selective_channel.png`)

| s | A (all) | B (smile-pure 110) | C (smile-pure − leak 105) |
|---:|---|---|---|
| −3 | 어린 남자 | 동일 남성, 무표정 | 동일 남성, 무표정 |
|  0 | 무표정 남성 | 무표정 남성 | 무표정 남성 |
| +2 | 시작되는 gender drift | **동일 남성, 약한 미소** | **동일 남성, 약한 미소** |
| +3 | 미소 짓는 **여성 (blonde)** | 약한 미소, 동일 남성 | 약한 미소, 동일 남성 |

→ **Selective injection 동작 확인**. Row B/C는 gender 안정, smile 변화는
약함. B와 C 차이는 시각적으로 미세 (leakiest 5개 제거가 미미한 영향).

### Trade-off: signal magnitude vs disentanglement

110 channel 사용은 512 사용보다 inject magnitude가 작아짐. Phase 16b
(`scripts/18_smile_pure_boost.py`)로 s를 ±6까지 확장:

`figures/exp12b_smile_pure_boost.png`:

- s ∈ [−4, +4]: gender 안정, smile만 control
- s = +6: **다시 여성으로 drift** — selective mask도 강한 inject에서는
  새어버림

해석:
- **smile-pure channel set이 절대 isolation은 아님**. 큰 inject은
  downstream 비선형성 (LayerNorm, residual coupling)을 통해 gender
  특성을 활성화할 수 있음
- **Clean editing range: |s| ≤ 4** 정도. 이 범위 내에선 깔끔한
  disentanglement 성립
- 실용적으로는 충분 (Phase 11에서 본 큰 변화는 |s| ≤ 3에서 일어남)

### 함의

Diffusion model bottleneck의 attribute structure는 **rank-1 channel 분리
모델로 잘 근사**되지만 perfect는 아님. 채널 단위 selective injection은:
1. 기능: gender-stable smile editing (|s| ≤ 4)
2. 한계: 큰 inject 또는 다중-attribute joint editing에서는 추가 안전망
   필요 (예: 동시에 gender 채널을 anti-inject로 활성화 보정)

다음 후보 (Phase 17): soft-purity weighting, anti-injection, attribute
direction이 시간축에 걸쳐 stable한지 검증 (t=450 categorization이 다른
t에서도 유효한가).

---

## 6g. E-GBAS polytope sampling은 diffusion에서 비효과적 (Phase 17)

E-GBAS의 GB-RRT (sign-pattern polytope 안에서 sampling)를 diffusion
h-space로 이식. 4가지 변형 모두 실패 (상세: `docs/11_egbas_analog.md`).

| 변형 | 제약 | acceptance | decoded diversity |
|---|---|---:|---|
| single-t | 512 ch @ t=500 | 55% | ε-ball과 동일 |
| selective | 110 smile-pure @ t=500 | 78% | ε-ball과 동일 |
| trajectory | 512 ch @ 5 timestep | 4.7% | ε-ball과 동일 |

GB-RRT는 모든 변형에서 polytope를 정확히 유지 (0 sign flip) 하지만,
decoded image는 ε-ball (polytope 무시) 과 구별되지 않음.

**왜**: GAN은 single feedforward pass라 activation pattern이 곧
piecewise-linear region을 정의 → 같은 region = 비슷한 output. Diffusion은
image가 50 step trajectory로 만들어지므로 한 timestep의 sign pattern은
최종 image를 제어하지 못함.

→ **방법론적 결론**: feedforward network의 activation-polytope paradigm
(E-GBAS/RDR/SplineCam)은 iterative diffusion으로 직접 transfer되지 않음.

---

## 6h. Speciation time — smile attribute의 commit zone (Phase 18-19)

Trajectory-aware boundary (T6) 의 첫 실증. 상세: `docs/12_*.md`.

### Speciation (Phase 18, held-out d')

| t | 50 | 450 | 550 | 650 | 750 | 850 | 950 |
|---|---:|---:|---:|---:|---:|---:|---:|
| d'_h | 1.85 | 1.70 | 1.45 | 0.98 | 0.50 | 0.08 | 0.02 |

- d'_h sigmoid 형태, **steepest rise t ∈ [650, 750]** — smile speciation zone
- **t=500은 speciation point 아님** — 이미 commit 거의 완료. 진짜
  decision-relevant timestep은 t ≈ 700
- held-out d' = 1.7 — Phase 10의 d'=4.23은 fit/test circularity로
  부풀려진 값이었음. 정직한 separation ≈ 1.7

### Score-Jacobian concentration (Phase 19)

`∂ε/∂x_t` top-4 singular value, σ1²-share: 대부분 t에서 0.25 (degenerate),
**t=760에서만 spike 0.41**. Speciation zone과 일치.

→ **t ≈ 700이 smile attribute의 commit zone**: h-space separability 급변
+ score-Jacobian collapse. trajectory-aware boundary의 첫 증거.

## 6i. Bifurcation은 없다, 그러나 editing leverage는 있다 (Phase 20-21)

상세: `docs/13_*.md`.

### Bifurcation test (Phase 20)
t≈700에서 x_t를 smile 방향으로 perturbation → smile_score(s) 는 **smooth
sigmoid ramp** (s=−6: −289 → s=+6: +236). Sharp step 없음. Random 방향
control은 flat → 방향은 의미 있으나 전이는 연속.

→ **t≈700에 sharp decision boundary (basin-flip) 없음**. Speciation은
attribute가 encode되는 지점이지 knife-edge가 아님.

### Speciation-window editing (Phase 21)
Δh injection을 timestep window로 제한:

| Window | steps | 효과 |
|---|---:|---|
| all | 50 | 강함 |
| speciation [600,800] | 11 | all과 동등 |
| early [350,550] | 10 | 약함 |

→ **speciation window 11 step = all 50 step**. editing leverage는 step
수가 아니라 위치(t≈700). 우리가 줄곧 쓴 t=500은 약한 zone이었음.

### "Polytope 구할 수 있나" 최종 답
- **Hard polytope: 없음** — Phase 20이 sharp boundary 부재 확인
- **Soft speciation region: 있음** — t≈700 연속 전이 zone, 경계가
  hyperplane이 아니라 smooth sigmoid. 닫힌 수식 없으나 editing leverage
  위치는 특정 가능

---

## 6j. 한 점에서 unsupervised concept direction — h-Jacobian으로 가능 (Phase 22-23)

"한 점에서 sampling으로 concept direction을 찾을 수 있나?" 에 대한 답.
상세: `docs/14_*.md`, `docs/15_*.md`.

| 방법 | 결과 |
|---|---|
| Polytope 샘플링 (Phase 17) | ❌ |
| ε-output Jacobian `∂ε/∂x` (Phase 22) | ❌ near-isotropic (σ1/σ5≈1.01), v1만 entangled 변화 |
| **h-space Jacobian `∂h/∂x` (Phase 23)** | ✅ σ1/σ5=1.66, 5방향 모두 의미 있는 semantic 변화 |
| Supervised Δh (Phase 11-16) | ✅✅ 단일 attribute 깨끗 |

**핵심**: operator 선택이 결정적. ε-output Jacobian은 near-isotropic이라
concept이 안 나오지만, **h-space Jacobian (bottleneck feature 대상)** 은
구조가 있어 top singular vector가 의미 있는 방향을 줌 (Park et al.
2302.12469 방법). 단 그 방향들은 gender/identity/expression이 섞인
**entangled** 상태 — 깨끗한 단일 axis는 여전히 supervision이 유리.

**여러 방향 (Phase 24)**: 한 점에서 top-12 h-Jacobian singular vector를
뽑으면 σ가 매끄럽게 감소 (41.8 → 19.8, sharp cutoff 없음) — local
semantic subspace가 최소 12차원. 12개 방향 모두 distinct한 semantic
변환을 생성하나 어느 것도 smile/gender와 깨끗이 정렬 안 됨 (max |cos|
0.18). → **한 점에서 여러 방향 추출 가능하나, singular basis는 semantic
subspace의 임의 정규직교 기저일 뿐 concept-aligned 기저가 아님.**

**깨끗한 axis 추출 (Phase 25-26)**:

| 방법 | 결과 |
|---|---|
| Subspace + supervision hybrid (Phase 25) | ✅ raw supervised와 동등한 editing |
| ICA rotation (Phase 26) | ❌ best cos 0.18→0.20, 미미 |

- Jacobian 12D subspace는 supervised smile 방향의 **33%** 포함 (random
  subspace는 0.8% — 40배 차이). 단 capture ratio 0.33인데도 projection을
  renormalize해 inject하면 **raw supervised와 동등한 smile editing** —
  잡힌 33%가 의미적으로 effective한 성분이고 놓친 67%는 모델이 반응 안
  하는 방향
- ICA rotation은 통계적 독립성을 최적화 → "smile/gender 정렬"과 불일치
  → disentangle 미미

→ **결론**: unsupervised h-Jacobian으로 semantic subspace는 찾되, 그 안의
깨끗한 단일 concept axis는 **minimal supervision (label-derived direction의
projection)** 으로 뽑는 하이브리드가 정답. 순수 unsupervised rotation
(ICA) 으로는 부족.

---

## 7. 핵심 결론

1. **2D plane construction은 exact**. ReLU/SiLU 같은 활성함수 종류는 결과에
   영향 없음 (sign(SiLU(z)) = sign(z)).
2. **Reduction 선택은 결정적**: spatial-max 금지, mean/median 사용.
3. **Top-K-balanced 선택은 시각적 artifact를 만든다** — radial 패턴은
   선택 편향 + 모델의 rank-2-ish 구조가 결합된 결과.
4. **Per-neuron boundary**는 방향·위치 모두 uniform. RDR/SplineCam 스타일의
   "진짜" 시각화는 이 단위에서 시작해야 함.
5. **채널 내부**에는 의미 있는 방향 편향이 있음. 채널을 선택해 그 64개
   line을 보는 게 본질적 정보 단위 후보.
6. **mid_block 단독 h 교체는 약한 editing**. 강한 attribute control은
   encoder skip 정보를 함께 manipulate해야 가능할 가능성.

---

## 8. 후속 방향 (우선순위 순)

1. **시간축 sweep** — t ∈ {200, 500, 700}에서 channel 내부 방향성이 변하는지
2. **Attribute-paired anchors** — 단순 random 3개가 아닌, 동일 attribute
   대비 pair (예: 안경 vs 안경 없음)로 plane을 구성하면 boundary 의미가
   attribute에 directly 결부되는지
3. **채널별 main-axis clustering** — 각 채널의 line bundle main direction을
   추출 → channel grouping → semantic concept 후보 발견
4. **Skip connection도 함께 manipulate**하는 decoder 변형으로 editing 강화
5. **RDR 논문 figure 1**과 시각적 비교 — 우리 결과의 fragmentation이
   기존 결과와 어떻게 다른지

각 항목은 별도 ablation script + figure로 추가될 예정.
