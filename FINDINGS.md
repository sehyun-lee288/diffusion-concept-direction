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

- **(A) attribute-paired anchor 직진 plan은 dead.** Supervised direction
  도 시각적 변화 없음 → unsupervised plane 분석은 더 약할 수밖에.
- **다음 step = skip-aware injection** (FINDINGS §8 item 4):
  encoder skip도 함께 manipulate해 mid_block과 일관된 정보 흐름을 만듦.
  구체 방법: skip activations에서도 Δskip을 supervised로 학습해 동시
  inject (mid_block injection + skip injection의 짝).
- 또는 Asyrp 식의 **iterative editing** — t를 변화시키며 여러 step에
  걸쳐 점진 inject.

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
