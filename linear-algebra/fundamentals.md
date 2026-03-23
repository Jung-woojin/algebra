# 컴퓨터비전/딥러닝 연구자를 위한 선형대수 실전 가이드

## 1. 이 문서의 목적
이 문서는 선형대수를 "시험 과목"이 아니라 "논문 읽기, 아키텍처 설계, 학습 안정화, 성능 디버깅"의 언어로 재정렬한다.  
핵심은 다음 3가지를 동시에 잡는 것이다.
- 직관: 이 개념이 어떤 기하/정보 구조를 표현하는가
- 수식: 논문에서 등장하는 표현을 바로 읽을 수 있는가
- 구현 감각: PyTorch 코드에서 shape/연산/수치 안정성으로 이어지는가

## 2. 왜 이 개념이 비전/딥러닝에 중요한가
- CNN: convolution, 1x1 conv, attention projection은 선형 변환의 조합이다.
- Representation learning: embedding 공간의 구조(거리, 각도, rank, subspace)가 성능을 좌우한다.
- Optimization: gradient, Jacobian, Hessian은 손실 지형과 학습 안정성을 설명한다.
- Metric learning: 코사인 유사도, Mahalanobis 거리, 정규화는 선형대수 그 자체다.
- Transformer in Vision: Q/K/V projection, low-rank 근사, whitening이 핵심 도구다.
- 3D vision: pose/geometry는 회전행렬, 고유값 분해, SVD 기반 추정 위에서 작동한다.

## 3. 핵심 개념

### 3.1 System of Linear Equations
- 핵심 아이디어: 여러 제약식을 동시에 만족하는 해를 찾는 문제 `Ax=b`.
- 연구자 관점: 해의 존재/유일성은 rank 조건으로 판단하고, 과결정 시스템은 최소제곱으로 푼다.
- 엔지니어 관점: 직접 역행렬보다 `torch.linalg.lstsq`, `solve`를 사용하고 conditioning을 항상 본다.
- CV 예시: 카메라 보정/pose 추정에서 선형화된 방정식 시스템을 반복적으로 풂.
- 논문 등장 포인트: PnP, bundle adjustment 초기화, normal equation `A^T A x = A^T b`.

### 3.2 Vector Spaces, Basis, Dimension
- 핵심 아이디어: 데이터/특징이 놓이는 "가능한 방향의 집합"과 그 최소 생성 집합.
- 연구자 관점: representation의 유효 차원(intrinsic dimension)과 subspace 구조를 분석한다.
- 엔지니어 관점: feature dimension을 늘리는 것이 항상 정보량 증가를 뜻하지 않음을 염두에 둔다.
- CV 예시: 얼굴 임베딩에서 실제 분별 정보는 전체 차원보다 훨씬 낮은 subspace에 집중.
- 논문 등장 포인트: latent space disentanglement, intrinsic rank 분석, manifold 가정.

### 3.3 Linear Independence
- 핵심 아이디어: 벡터들이 서로 중복 없이 새로운 정보를 제공하는가.
- 연구자 관점: collapse 현상(표현 중복)을 독립성 저하로 해석하고 regularizer를 설계한다.
- 엔지니어 관점: 배치 feature의 공분산/상관행렬로 중복도를 모니터링한다.
- CV 예시: self-supervised learning에서 projector 출력이 한 방향으로 쏠리면 성능 급락.
- 논문 등장 포인트: redundancy reduction, decorrelation loss, whitening objectives.

### 3.4 Matrix Multiplication as Transformation
- 핵심 아이디어: 행렬 곱은 "회전/스케일/shear/투영" 같은 선형 변환의 합성.
- 연구자 관점: layer stack을 연속 선형 변환 + 비선형으로 보고 표현력과 안정성을 분석한다.
- 엔지니어 관점: `W2(W1x)`는 연산자 합성이고, shape 추적이 수학 오류 예방의 1순위다.
- CV 예시: 1x1 convolution은 채널 공간 선형 혼합, attention의 QK^T도 변환 합성.
- 논문 등장 포인트: linear probe, projection head, token mixing matrix 해석.

### 3.5 Rank and Null Space
- 핵심 아이디어: rank는 정보 차원, null space는 입력이 소거되는 방향.
- 연구자 관점: 모델이 어떤 정보를 보존/폐기하는지 rank 스펙트럼으로 진단한다.
- 엔지니어 관점: weight matrix의 effective rank를 추적해 과압축/붕괴를 잡는다.
- CV 예시: 저조도 복원에서 복원 연산이 특정 주파수 성분을 null space로 보내 디테일 손실 유발.
- 논문 등장 포인트: low-rank adapter, bottleneck 설계, spectral regularization.

### 3.6 Orthogonality and Projection
- 핵심 아이디어: 직교는 정보 중복 최소화, projection은 관심 subspace로의 분해.
- 연구자 관점: nuisance factor 제거를 projection으로 모델링한다.
- 엔지니어 관점: Gram-Schmidt보다 `QR/SVD` 기반 구현을 선호해 수치 안정성 확보.
- CV 예시: 배경/조명 성분을 orthogonal subspace로 분리해 식별 성능 개선.
- 논문 등장 포인트: orthogonal constraint, projection head, subspace decomposition.

### 3.7 Eigenvalues and Eigenvectors
- 핵심 아이디어: 변환 후 방향은 유지되고 크기만 변하는 축.
- 연구자 관점: 동역학/학습 안정성(수렴, exploding/vanishing)의 핵심 지표로 본다.
- 엔지니어 관점: 스펙트럴 반경을 통해 반복 업데이트 안정 구간을 점검한다.
- CV 예시: diffusion/graph-based vision에서 propagation 연산의 안정성 분석.
- 논문 등장 포인트: spectral norm bounds, graph Laplacian eigen-spectrum.

### 3.8 Diagonalization
- 핵심 아이디어: 복잡한 선형변환을 고유기저에서 축별 독립 스케일링으로 단순화.
- 연구자 관점: 연산 해석/복잡도 감소/이론 분석에 유리한 좌표계를 찾는 문제.
- 엔지니어 관점: 완전 대각화 불가한 경우가 많아 SVD/Schur로 우회한다.
- CV 예시: 공분산 구조를 대각화해 decorrelation/whitening 수행.
- 논문 등장 포인트: second-order method 근사, preconditioning 분석.

### 3.9 SVD
- 핵심 아이디어: 임의 행렬을 회전-스케일-회전(`UΣV^T`)으로 분해.
- 연구자 관점: low-rank 구조, 노이즈 분리, 표현 차원 축약의 표준 도구.
- 엔지니어 관점: `torch.linalg.svd`는 비싸므로 randomized/truncated SVD를 고려.
- CV 예시: LoRA, 압축, 배경/객체 분리, fundamental matrix 추정.
- 논문 등장 포인트: low-rank adaptation, nuclear norm, robust PCA류 방법.

### 3.10 Positive Definite Matrices
- 핵심 아이디어: `x^T A x > 0`를 만족하는 곡면; 에너지/분산/곡률을 정의.
- 연구자 관점: 공분산, metric tensor, Hessian 근사에서 SPD 구조를 활용한다.
- 엔지니어 관점: Cholesky 분해 가능 여부로 수치적 건강 상태를 빠르게 확인.
- CV 예시: Mahalanobis metric learning에서 SPD 행렬이 클래스 분별 거리 정의.
- 논문 등장 포인트: Gaussian modeling, covariance pooling, Riemannian optimization.

### 3.11 Quadratic Forms
- 핵심 아이디어: `x^T A x`는 방향별 에너지/패널티를 측정하는 함수.
- 연구자 관점: regularization을 단순 L2에서 방향가중 penalty로 확장한다.
- 엔지니어 관점: loss 항을 quadratic으로 보면 gradient/Hessian 구조를 예측 가능.
- CV 예시: optical flow smoothness, graph regularization.
- 논문 등장 포인트: Tikhonov regularization, energy minimization framework.

### 3.12 Norms and Distances
- 핵심 아이디어: 크기(norm)와 유사도(distance)가 representation의 학습 목표를 규정.
- 연구자 관점: L2, cosine, Mahalanobis 선택은 귀납 편향 그 자체다.
- 엔지니어 관점: feature normalize 여부가 metric learning 성능에 직접 영향.
- CV 예시: face recognition에서 cosine margin loss.
- 논문 등장 포인트: contrastive/triplet/arcface 계열 손실 설계.

### 3.13 Matrix Factorization
- 핵심 아이디어: 복잡한 행렬을 의미 있는 작은 요인들로 분해.
- 연구자 관점: "무엇이 데이터 생성요인인가"를 factor로 해석한다.
- 엔지니어 관점: 파라미터 절감, 계산량 축소, 모듈화에 바로 적용.
- CV 예시: attention/kernel의 low-rank factorization으로 모바일 추론 가속.
- 논문 등장 포인트: CP/Tucker, low-rank conv, adapter decomposition.

### 3.14 Matrix Calculus Basics
- 핵심 아이디어: 벡터/행렬 변수에 대한 미분 규칙.
- 연구자 관점: 손실 설계를 gradient flow 관점에서 읽고 안정성 조건을 점검.
- 엔지니어 관점: autograd를 믿되, 차원과 transpose 규칙을 수동 검증한다.
- CV 예시: custom loss 구현 시 broadcasting 실수로 잘못된 gradient가 전파되는 문제.
- 논문 등장 포인트: 부록의 gradient 유도식, closed-form update.

### 3.15 Jacobian / Hessian Intuition
- 핵심 아이디어: Jacobian은 국소 선형화, Hessian은 곡률/조건수 정보.
- 연구자 관점: representation 민감도, sharpness, generalization 관계를 본다.
- 엔지니어 관점: Hessian full 계산 대신 trace, top-eigen, Fisher 근사를 사용.
- CV 예시: adversarial robustness에서 입력 Jacobian norm 제어.
- 논문 등장 포인트: sharpness-aware training, second-order approximation.

### 3.16 PCA and Low-rank Approximation
- 핵심 아이디어: 분산이 큰 축을 남기고 작은 축을 버려 정보 압축.
- 연구자 관점: 데이터 통계 구조를 분석해 inductive bias를 찾는다.
- 엔지니어 관점: feature 시각화/노이즈 제거/메모리 절감 파이프라인에 즉시 적용.
- CV 예시: ViT feature PCA 시각화로 클래스 분리 정도 확인.
- 논문 등장 포인트: self-supervised feature 분석, compression/pruning 사전분석.

### 3.17 Optimization and Gradient-based Learning Connection
- 핵심 아이디어: 학습은 고차원 함수 위에서 선형근사(1차/2차)를 반복하는 과정.
- 연구자 관점: preconditioner, curvature, spectral bias가 최적화 경로를 결정.
- 엔지니어 관점: learning rate, weight decay, normalization은 선형대수적 스케일 제어 장치.
- CV 예시: 배치 크기/학습률 스케일링 실패 시 loss surface의 ill-conditioning이 노출.
- 논문 등장 포인트: Adam/SGD 해석, natural gradient, K-FAC류 근사.

## 4. 수식 직관
- 선형 시스템: `Ax=b`는 "A가 만든 공간 위에서 b를 맞출 수 있나?"의 문제다.
- 투영: `P = A(A^T A)^{-1}A^T`는 벡터를 `col(A)`로 떨어뜨리는 연산.
- 고유분해: `Av = λv`는 변환의 고정 방향과 증폭률.
- SVD: `A = UΣV^T`는 입력/출력 기저를 분리해 정보량(`Σ`)을 노출.
- 양정정부호: `x^T A x > 0`이면 에너지 해석 가능, 최적화에서 안정적.
- Jacobian: `J = ∂f/∂x`는 작은 변화가 출력에서 어떻게 증폭되는지 보여준다.
- Hessian: `H = ∂^2L/∂θ^2`는 곡률; 큰 고유값은 가파른 방향, 작은 값은 평평한 방향.

## 5. 딥러닝/컴퓨터비전 연결
- CNN: convolution도 로컬 선형연산이며 BN/ReLU와 결합해 비선형 표현력 확보.
- Metric learning: distance choice가 곧 embedding geometry.
- Transformer: Q/K/V projection은 선형변환, attention map은 Gram 구조와 밀접.
- 3D vision: epipolar/pose/triangulation은 선형화 + 최소제곱 + SVD 반복.
- Representation learning: collapse 방지는 rank/orthogonality/covariance 제어 문제.

## 6. 자주 헷갈리는 포인트
- rank가 크면 항상 좋은가: 아니다. task-relevant signal 대비 noise rank도 함께 증가할 수 있다.
- 역행렬을 항상 구해야 하나: 아니다. 대부분 solve/lstsq가 더 안정적이다.
- 고유분해와 SVD는 같은가: 대칭행렬에서는 연결되지만 일반행렬에서는 다르다.
- Jacobian/Hessian은 너무 비싸서 무쓸모인가: full 계산은 비싸지만 근사 통계는 매우 유용하다.
- PCA 성능 저하 = 나쁜 방법인가: 목적이 압축/시각화/노이즈 제거라면 유효하다.

## 7. 작은 예제 또는 numpy/pytorch 실험 아이디어
```python
# 1) Effective rank 추적
import torch
W = torch.randn(512, 512)
s = torch.linalg.svdvals(W)
effective_rank = torch.exp(-(s/s.sum()) * torch.log((s/s.sum()) + 1e-12)).sum()
print("effective rank:", float(effective_rank))

# 2) Projection 실험
A = torch.randn(128, 16)
P = A @ torch.linalg.inv(A.T @ A) @ A.T
x = torch.randn(128)
x_proj = P @ x
print("projection residual norm:", float(torch.norm(x - x_proj)))

# 3) Jacobian norm (입력 민감도) 측정
model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))
inp = torch.randn(1, 32, requires_grad=True)
out = model(inp).sum()
out.backward()
print("||dout/dx||:", float(inp.grad.norm()))
```

## 8. 논문 읽기 연결 포인트
- "spectral", "orthogonal", "rank", "subspace", "conditioning", "curvature" 키워드가 나오면 선형대수 핵심 구간이다.
- Method 섹션에서 projection/factorization을 쓰면 보통 학습 안정성 또는 계산 효율이 목적이다.
- Appendix의 gradient 유도식은 구현에서 shape mismatch를 예방하는 체크리스트로 사용한다.
- 3D vision 논문에서 SVD/eigendecomposition은 종종 "해의 기하학적 제약"을 만족시키는 도구다.

## 9. GitHub에 같이 링크할 후속 문서
권장 분할 구조:
- `linear-algebra/fundamentals.md`: 전체 개념 지도 + CV 연결
- `linear-algebra/eigens-and-svd.md`: 고유분해/SVD 심화, low-rank 설계 패턴
- `linear-algebra/matrix-calculus.md`: matrix calculus, Jacobian/Hessian, autograd 검증 패턴
- `linear-algebra/optimization-geometry.md`: conditioning, preconditioning, curvature 기반 학습 안정화
- `linear-algebra/metric-and-representation.md`: norm/distance, metric learning, representation geometry
- `linear-algebra/vision-geometry.md`: 3D vision과 선형대수(카메라/pose/triangulation)

## 10. 핵심 질문 5개
1. 지금 모델의 feature 공간은 어떤 subspace를 학습하고 있고, 불필요한 null space는 무엇인가?
2. 성능 병목이 표현력 부족(rank 부족)인지, 최적화 불안정(conditioning 문제)인지 어떻게 구분할 것인가?
3. 내가 쓰는 거리 함수(L2/cosine/Mahalanobis)는 task의 의미론과 맞는가?
4. Jacobian/Hessian 근사 통계를 보면 현재 학습률/정규화 설정이 타당한가?
5. 이 논문의 핵심 기여를 선형변환/분해/투영 관점으로 재해석하면 무엇이 남는가?
