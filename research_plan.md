# Diffusion Model H-Space에서의 Decision Boundary 시각화 및 분석

---

## 1. Research Goal

Deep generative model의 내부 구조를 이해하는 것은 생성 메커니즘을 해석하고, concept을 제어하는 데 핵심적이다.
GAN이나 classifier 기반 모델에서는 neuron activation pattern의 zero-crossing으로 정의되는 decision boundary를 직접 시각화하는 연구들이 존재한다.
그러나 diffusion model에 대한 유사한 분석은 거의 없다.

본 연구의 목표는 **DDPM의 U-Net bottleneck (h-space)에서 정의되는 activation boundary를 2D 평면 위에 시각화하고, 그 경계가 생성 이미지의 semantic attribute와 어떻게 대응하는지 분석하는 것**이다.
장기적으로는 label이나 별도 classifier 없이, boundary 구조만으로 concept direction을 발견하는 방법론으로 확장하는 것을 목표로 한다.

---

## 2. Research Questions

**RQ1 (핵심):** Diffusion model U-Net의 bottleneck layer에서 정의되는 SiLU pre-activation sign pattern boundary가, 생성 이미지의 semantic attribute (예: 머리 색, 성별, 미소)와 어떻게 대응하는가?

**RQ2:** 특정 timestep (t=500)에서 h-space boundary를 분석하는 것이 의미 있는가? 즉, 이 시점의 activation pattern이 final output의 attribute를 충분히 설명하는가?

**RQ3:** 3개의 anchor image가 span하는 2D 평면 위에서, boundary로 나뉜 region들이 서로 다른 semantic attribute에 대응하는 구조를 가지는가?

**RQ4 (장기):** Label 없이 boundary 구조만으로 concept direction (예: Asyrp의 Δh)을 발견할 수 있는가?

---

## 3. Related Work 분석

### 3.1 E-GBAS (Jeon et al., 2019 — arXiv:1912.05827)

**핵심 아이디어:** GAN의 latent space는 generative boundary (각 neuron의 pre-activation = 0 인 hyperplane)로 분할된다. 같은 region 안의 latent vector는 동일한 activation pattern → 동일한 attribute를 가진 이미지를 생성한다.

**방법:**
- Bernoulli Mask Optimization (BerOpt): query z₀에 대해 output을 유지하면서 최소 개수의 경계만 남기는 SSGBS를 최적화로 구함
- GB-RRT: SSGBS로 정의된 region 내부를 탐색하며 샘플 수집

**Decision Boundary 시각화:** DCGAN-MNIST의 latent space가 2D이기 때문에 boundary를 직접 그릴 수 있음. 고차원 모델 (PGGAN-CelebA)에서는 경계를 시각화하지 않음.

**본 연구와의 관계:** 핵심 개념 (activation sign pattern = generative region)을 diffusion model의 h-space로 이식하는 것이 출발점. 단, diffusion은 iterative denoising이고 U-Net은 SiLU를 사용하므로 직접 적용 불가.

### 3.2 SplineCam (2302.12828)

**핵심 아이디어:** ReLU network는 continuous piecewise linear (CPWL) 함수이므로, input space를 affine region으로 exact하게 분할할 수 있다. Decision boundary는 output layer의 classification hyperplane을 각 region에 투영한 것.

**방법:**
- 임의의 bounded 2D domain P를 지정
- 각 layer의 hyperplane이 P와 교차하는 지점을 계산
- BFS로 closed region (cycle)을 추출
- 각 region의 predicted class를 평가하여 boundary를 그림

**본 연구와의 관계:** Exact boundary 계산 방법론의 참조. 단, diffusion U-Net은 SiLU (smooth) + attention을 사용하므로 hard zero-crossing 기반의 exact partition은 적용 불가. Sign threshold를 이용한 근사 방법으로 대체.

### 3.3 RDR / Configuration Distance (2312.17285)

**핵심 아이디어:** 두 input 사이의 "Configuration Distance"를 activation sign vector 간의 Hamming distance로 정의. Relaxed Decision Region (RDR)은 query와 동일한 sign pattern을 가지는 feature space 내의 점들의 집합.

**시각화 방법:**
- 특정 layer (예: VGG19의 12번째 layer)의 feature space에서 3개의 instance를 지나는 2D 평면을 정의
- 해당 평면 위에서 각 neuron의 pre-activation = 0 인 line을 그림 → boundary web 형성
- Query의 RDR (동일한 sign pattern 영역)을 하이라이트

**본 연구와의 관계:** **2D 평면 정의 방법 (3개의 anchor point 사용)을 직접 차용한다.** Diffusion U-Net의 h-space에서 동일한 방식으로 2D 평면을 span하고, SiLU pre-activation의 sign boundary를 그린다.

### 3.4 Asyrp (Kwon et al., 2023)

**핵심 아이디어:** Diffusion model U-Net의 bottleneck feature (h-space)는 semantic latent space로서 기능한다. 특정 timestep (t ≈ 500–700)에서 h에 방향 벡터 Δh를 더하면 생성 이미지의 attribute가 disentangled하게 변한다.

**본 연구와의 관계:** h-space가 semantic 정보를 가장 잘 담고 있는 공간임을 지지하는 근거. h-space에서의 boundary 분석이 의미 있음을 이론적으로 뒷받침.

### 3.5 DDPM / DDIM (Ho et al. 2020; Song et al. 2021)

- Forward process: q(x_t | x_{t-1}) — 점진적으로 Gaussian noise 추가, t=0이 clean, t=T=1000이 pure noise
- Reverse process: x_T → x_0 denoising
- DDIM: deterministic inversion 가능 → 주어진 이미지를 x_500으로 invert 가능

---

## 4. Design Decision

| 항목 | 선택 | 이유 |
|---|---|---|
| **Model** | DDPM on CelebA (256×256) | 고품질 unconditional 모델, attribute가 풍부하여 분석에 적합 |
| **Timestep** | t = 500 | Asyrp에서 semantic 정보가 가장 풍부한 intermediate zone |
| **H-space 정의** | U-Net bottleneck (가장 작은 spatial resolution의 feature map) | Semantic content가 가장 압축된 layer |
| **Boundary 정의** | Bottleneck의 SiLU pre-activation sign (각 channel) | ReLU zero-crossing의 soft 대안; sign 변화 = activation pattern 변화 |
| **2D 평면 정의** | 3개의 anchor image → h-space에서 span | Paper 3 (RDR)과 동일한 방식, label 불필요 |
| **Grid decoding** | DDIM 20-step, sparse grid | 계산 비용 절감 (50×50 full grid = 2500 × 500 step은 비현실적) |

### 구체적 2D 평면 정의

```
h_1, h_2, h_3: 3개 anchor image의 t=500 bottleneck feature (flatten vector)

origin = h_1
u = (h_2 - h_1) / ||h_2 - h_1||          # 첫 번째 basis
v' = h_3 - h_1
v = (v' - (v'·u)u) / ||v' - (v'·u)u||   # u에 직교하는 두 번째 basis

grid point: h(α, β) = h_1 + α·u + β·v
```

### Boundary 정의 (SiLU sign)

SiLU(x) = x · σ(x)이므로 pre-activation x의 부호가 출력의 방향을 결정한다.
각 channel c에 대해:

```
s_c(α, β) = sign(pre_SiLU_c(h(α, β)))  ∈ {-1, +1}
```

Configuration vector: **s(α, β) = [s_1, s_2, ..., s_C]**

Boundary: 인접한 grid point 사이에서 **s가 변하는 위치**

---

## 5. Toy Experiment 1: H-Space 2D Boundary 시각화

### 목표

DDPM CelebA에서 t=500의 h-space를 3개의 anchor image로 span한 2D 평면 위에 SiLU sign boundary를 시각화하고, 각 region이 의미 있는 semantic attribute에 대응하는지 확인한다.

### 실험 절차

**Step 1. Anchor image 선택**
- CelebA test set에서 3개 이미지 선택
- 첫 번째 실험: 서로 다른 attribute를 가진 이미지 (예: 금발 여성 / 갈색머리 남성 / 미소 짓는 사람)

**Step 2. DDIM Inversion**
- 각 anchor image를 DDIM으로 x_500으로 invert

**Step 3. H-vector 추출**
- x_500을 U-Net에 입력, t=500 condition으로 forward pass
- Bottleneck feature를 h_1, h_2, h_3으로 저장

**Step 4. 2D Grid 정의 및 sign pattern 계산**
- α ∈ [-1.5, 1.5], β ∈ [-1.5, 1.5] 범위를 50×50 grid로 sampling
- 각 grid point h(α, β)에 대해 bottleneck SiLU pre-activation의 sign vector 계산
- 인접 grid point 간 Hamming distance가 0이 아닌 위치에 boundary 표시

**Step 5. Boundary 시각화**
- 2D 평면 위에 boundary web을 그림
- Region별로 서로 다른 색으로 채워 activation pattern region을 시각화
- Anchor point h_1, h_2, h_3의 위치를 표시

**Step 6. Sparse decoding으로 semantic 검증**
- Grid 위 representative point 9~16개 선택 (각 major region에서 1개)
- 각 점에서 DDIM 20-step denoising으로 이미지 생성
- 생성된 이미지의 attribute 확인

### 기대 결과

- Boundary가 random하지 않고 structured된 region을 형성할 것
- 동일 region 내 decoded image들이 유사한 attribute를 공유할 것
- Anchor point 사이의 중간 region이 interpolated attribute를 보일 것

### 실패 기준 및 대응

| 실패 케이스 | 원인 추정 | 대응 |
|---|---|---|
| Boundary가 너무 많아 region 구분 불가 | Channel 수가 너무 많음 | PCA로 top-K channel만 사용, 또는 attention weight 기반 channel selection |
| 모든 grid point가 같은 sign pattern | Grid range가 너무 좁음 | α, β 범위 확장 |
| Decoded image들이 의미 없음 | t=500이 너무 noisy하여 decoding 정보 부족 | t=300으로 낮춰 재시도 |
| Region과 attribute 불일치 | h-space bottleneck이 아닌 다른 layer가 더 적합 | 여러 layer 비교 실험 |

---

## 6. 향후 방향 (Toy Experiment 이후)

1. **Channel selection**: 어떤 channel들이 의미 있는 boundary를 형성하는지 분석 → concept-relevant neuron 발견
2. **Timestep sweep**: t=200, 300, 500, 700에서 boundary 구조 비교 → semantic 정보가 가장 풍부한 timestep 탐색
3. **Unsupervised concept discovery**: Boundary로 나뉜 region들을 clustering → label 없이 concept 발견
4. **E-GBAS analogues**: GB-RRT와 유사하게, h-space의 boundary 안에서 탐색하며 consistent attribute sample 수집
