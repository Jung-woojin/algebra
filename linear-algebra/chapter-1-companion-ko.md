# Chapter 1 보조자료: 선형대수 직관과 실전

이 문서는 행렬과 벡터의 **기하학적 의미**와 **실제 계산**을 함께 이해할 수 있도록 만든 보조 자료입니다.  
이론과 실무를 연결하는 "다리" 역할을 합니다.

---

## 📚 사용 가이드

### 학습 방법
1. **핵심 질문** → 개념의 본질 이해
2. **직관 설명** → 머릿속에 그림 그리기
3. **계산 예제** → 손으로 직접 풀어보기
4. **시각화** → 그래프로 확인하기
5. **실전 활용** → 실제 문제에 적용

### 이 문서의 특징
- ✅ **기하학 중심**: 행렬을 "변환"으로 이해
- ✅ **실전 계산**: 이론과 실제 계산 연결
- ✅ **시각화 지원**: 그림으로 직관화
- ✅ **코드 예제**: NumPy/PyTorch 실습

---

## 1. 행렬-벡터 곱셈 Ax: 열의 선형결합

### 🎯 핵심 질문
**"Ax를 왜 행렬-벡터 곱이 아니라 '열의 조합'으로 봐야 하나?"**

### 📐 직관 설명
`A = [a₁ a₂ ... aₙ]`, `x = [x₁ x₂ ... xₙ]ᵀ`일 때:

```
Ax = x₁a₁ + x₂a₂ + ... + xₙaₙ
   = 열 1 에 x₁ 만큼 더하기
   + 열 2 에 x₂ 만큼 더하기
   + ...
```

**예시**:
```
A = [1  2]     x = [3]
    [0  1]         [4]

Ax = 3 × [1] + 4 × [2] = [ 3] + [ 8] = [11]
         [0]         [1]   [0]   [4]   [ 4]
```

### 💻 계산 실습 (NumPy)

```python
import numpy as np

A = np.array([[1, 2],
              [0, 1]])
x = np.array([3, 4])

# 방법 1: 행렬 - 벡터 곱
Ax1 = A @ x  # array([11, 4])

# 방법 2: 열의 선형결합
col1 = A[:, 0]  # array([1, 0])
col2 = A[:, 1]  # array([2, 1])
Ax2 = 3 * col1 + 4 * col2  # array([11, 4])

print(Ax1, Ax2)  # 동일 결과
```

### 🎨 시각화
```
기하학적 의미:
x = [3, 4]² → 벡터 x 가 basis 벡터들의 가중치
Ax         → 새로운 좌표계에서 점의 위치

   y
   ^
   |     /  (1,2)
   |    /
   |   /
   |  /
   | /
   |/
   +------> x
```

### 🔧 실전 활용
- **Computer Graphics**: 변환 행렬로 객체 회전/이동
- **Neural Networks**: fully connected layer = Ax
- **Signal Processing**: 필터 적용 = Convolution

### 📝 체크 문제
1. `rank(A) < n`이면 어떤 벡터 `x ≠ 0`에 대해 `Ax = 0`이 되는가?
2. `b`가 `Col(A)`에 있지 않으면 `Ax = b`는 어떤 상태인가?

---

## 2. 행렬 곱셈 AB: 변환의 합성

### 🎯 핵심 질문
**"AB의 기하학적 의미는 무엇인가?"**

### 📐 직관 설명
`B`가 먼저 변환 → `A`가 다시 변환

```
AB = [Ab₁ Ab₂ ... Abₙ]
     = A × (B 의 열 1), A × (B 의 열 2), ...
```

**예시**:
```
A = [1 0]      B = [2 0]
    [0 2]          [0 1]

AB = [1×2+0×0  1×0+0×1] = [2  0]
     [0×2+2×0  0×0+2×1]   [0  2]
```

### 💻 계산 실습

```python
import numpy as np

A = np.array([[1, 0],
              [0, 2]])
B = np.array([[2, 0],
              [0, 1]])

AB = A @ B
print(AB)  # [[2 0] [0 2]]

# 기하학적 의미:
# B: x축 2배, y축 그대로
# A: x축 그대로, y축 2배
# AB: x축 2배, y축 4배
```

### 🎨 시각화
```
단위 정사각형에 변환 적용:

   (0,1)           (0,1)           (0,4)
      |             |                 |
      |             |                 |
      |             |                 |
      +---> (1,0)   +---> (2,0)       +---> (2,4)

Before B        After B           After AB
```

### 🔧 실전 활용
- **Computer Graphics**: 여러 변환 행렬의 합성
- **Machine Learning**: layer 의 합성
- **Physics**: 좌표계 변환

### 📝 체크 문제
1. `rank(AB) <= min(rank(A), rank(B))`가 왜 성립하는가?
2. `AB = 0`이지만 `A ≠ 0`, `B ≠ 0`일 수 있는가?

---

## 3. 네 가지 기본 부분공간

### 🎯 핵심 질문
**"왜 열공간, 행공간, 영공간, 좌영공간 4 개를 함께 봐야 하나?"**

### 📐 직관 설명
입력과 출력 공간에서 **"살아남는 방향"**과 **"사라지는 방향"**

```
Row space (r)    Col space (r)
    ↓                   ↓
    |    Ax           |
    |  ----------->   |
Null space (n-r)      Left null (m-r)
(사라지는 방향)        (상대값 없는 방향)
```

### 💻 계산 실습

```python
import numpy as np
from scipy.linalg import null_space

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

r = np.linalg.matrix_rank(A)  # 랭크
n = A.shape[1]  # 열 개수

# 차원 계산
dim_row = r  # 행공차
dim_col = r  # 열공차
dim_null = n - r  # 영공차
dim_left_null = A.shape[0] - r  # 좌영공차

print(f"Rank: {r}")
print(f"Null space dim: {dim_null}")

# 실제 영공간 계산
N = null_space(A)
print("Null space basis:")
print(N)
```

### 🎨 시각화
```
A: 3×3, rank=2 인 경우

Input R³          Matrix A         Output R³
┌─────┐                         ┌─────┐
│  N  │  (1 차원 선)             │  C  │  (2 차원 평면)
│  │  │  ─────────────→         │  │  │
└─────┘                         └─────┘
Null space                     Col space

점선: 영공간 (Ax=0 이 되는 방향)
실선: 행공간 (살아남는 방향)
```

### 🔧 실전 활용
- **System의 해 존재성**: `b ∈ Col(A)` 여부 확인
- **과소/과대 결정**: `rank`, `n`, `m` 관계 확인
- **Stability 분석**: condition number 계산

### 📝 체크 문제
1. `A`가 5×3, rank=2 일 때 `N(A)`의 차원은?
2. `Ax=b`가 해를 갖기 위한 필요충분조건은?

---

## 4. 소거법과 A = LU 분해

### 🎯 핵심 질문
**"왜 LU 분해가 실전 계산의 핵심인가?"**

### 📐 직관 설명
소거법을 행렬 연산으로 표현:

```
A = L × U
    ↑   ↑
   하   상
  삼각  삼각
```

### 💻 계산 실습

```python
import numpy as np
from scipy.linalg import lu

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]])

# LU 분해
P, L, U = lu(A)

print("A = P^T L U")
print("L:")
print(L)
print("U:")
print(U)

# Ax=b 풀기
b = np.array([1, 2, 3])
# Ly = Pb
y = np.linalg.solve(L, P @ b)
# Ux = y
x = np.linalg.solve(U, y)

print("Solution x:", x)
print("Verification Ax:", A @ x)  # b 와 동일
```

### 🎨 시각화
```
소거 과정:

[2  1  1]        [1  0  0]        [2  1  1]
[4  3  3]  →     [l21 1  0]   →   [0  u22 u23]
[8  7  9]        [l31 l32 1]      [0  0   u33 ]

L (소거계수)      U (상삼각)
```

### 🔧 실전 활용
- **Linear system solving**: `Ax=b` 빠르게 풀기
- **Determinant 계산**: `det(A) = det(L) × det(U)`
- **Matrix inverse**: `A⁻¹ = U⁻¹ L⁻¹`

### 📝 체크 문제
1. 피벗이 0 일 때 행 교환이 필요한 이유는?
2. `A = LU`가 항상 존재하는가?

---

## 5. 직교행렬과 부분공간

### 🎯 핵심 질문
**"왜 직교행렬이 수치적으로 중요한가?"**

### 📐 직관 설명
직교행렬 `Q`: 길이와 각도 보존 → **회전/반사**

```
QᵀQ = I  ⇔  Q⁻¹ = Qᵀ
```

### 💻 계산 실습

```python
import numpy as np
from scipy.linalg import qr

# QR 분해로 직교행렬 생성
A = np.array([[1, 1],
              [1, 2],
              [1, 3]])

Q, R = qr(A)

print("Q (orthogonal):")
print(Q)
print("QᵀQ:")
print(Q.T @ Q)  # 단위행렬

# 길이의 보존
x = np.array([1, 2, 3])
print(f"||x|| = {np.linalg.norm(x)}")
print(f"||Qx|| = {np.linalg.norm(Q @ x)}")  # 동일
```

### 🎨 시각화
```
직교 변환의 특성:

Before:              After Q:
   y                    y'
   |                    |
   |        Q →         |
   +--- x      rotation +--- x'

길이 보존, 각도 보존
```

### 🔧 실전 활용
- **QR 알고리즘**: 고유값 계산
- **Least squares**: `min ||Ax - b||`
- **Stable transformations**: 수치 안정성 보장

### 📝 체크 문제
1. 직교행렬의 열벡터들은 어떤 관계인가?
2. `Q`가 직교행렬일 때 `det(Q)`의 가능한 값은?

---

## 6. 고윳값과 고유벡터

### 🎯 핵심 질문
**"고유벡터를 왜 찾는가?"**

### 📐 직관 설명
고유벡터: **방향은 유지되고 크기만 바뀜**

```
Av = λv
```

### 💻 계산 실습

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

# 고유값/벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# 확인
v = eigenvectors[:, 0]
lambda_val = eigenvalues[0]
print(f"A@v = {A @ v}")
print(f"lambda*v = {lambda_val * v}")  # 동일
```

### 🎨 시각화
```
고유벡터의 의미:

   y                    y
   |     A          |
   |     →         |
   +--- x           +--- x

고유벡터 v: 방향 그대로, 크기 λ배
           λ > 1: 확대
           0 < λ < 1: 축소
           λ < 0: 반전
```

### 🔧 실전 활용
- **System stability**: 고윳값 부호로 안정성 판별
- **Principal Components**: 공분산 행렬의 고유벡터
- **PageRank**: 웹 그래프의 고유벡터

### 📝 체크 문제
1. 대각화 가능 조건은?
2. `A`와 `P⁻¹AP`는 왜 같은 고유값을 가지는가?

---

## 7. 대칭 양의 정부호 (SPD) 행렬

### 🎯 핵심 질문
**"왜 SPD 가 최적화/ML 에서 중요한가?"**

### 📐 직관 설명
모든 방향에서 양의 에너지 = **볼록한 그릇**

```
xᵀSx > 0  (∀ x ≠ 0)
```

### 💻 계산 실습

```python
import numpy as np

# SPD 행렬 예시
A = np.array([[4, 2],
              [2, 3]])

# SPD 판별
print("Symmetric:", np.allclose(A, A.T))

# 고윳값 모두 양수?
eigenvalues = np.linalg.eigvalsh(A)  # 대칭 전용
print("Eigenvalues:", eigenvalues)
print("All positive:", np.all(eigenvalues > 0))

# 주축 판별
def is_spd(matrix):
    n = len(matrix)
    for k in range(n):
        minor = matrix[:k+1, :k+1]
        if np.linalg.det(minor) <= 0:
            return False
    return True

print("Principal minors:", is_spd(A))
```

### 🎨 시각화
```
이차형식 xᵀSx 의 등고선:

     y
     ^
   __|__
  /     \     타원형 등고선
 |   ●   |      중심에서 멀어질수록 값 증가
  \     /
   --+---> x

볼록함 (convex) = 국소최대 = 전역최대
```

### 🔧 실전 활용
- **Least squares**: `AᵀA`는 항상 SPD
- **Optimization**: Hessian 이 SPD 면 볼록함
- **Gaussian**: 공분산 행렬은 PSD

### 📝 체크 문제
1. `AᵀA`가 왜 항상 PSD 인가?
2. SPD 판별 3 가지 방법?

---

## 8. 특잇값 분해 (SVD)

### 🎯 핵심 질문
**"왜 모든 행렬이 SVD 로 분해 가능한가?"**

### 📐 직관 설명
임의의 선형변환을 **3 단계 회전/스케일**로 분해

```
A = U Σ Vᵀ

V: 입력 축 회전
Σ: 축별 스케일링
U: 출력 축 회전
```

### 💻 계산 실습

```python
import numpy as np

A = np.array([[3, 0],
              [4, 5]])

# SVD 분해
U, s, Vt = np.linalg.svd(A)
Sigma = np.diag(s)

print("U:")
print(U)
print("Σ:")
print(Sigma)
print("Vᵀ:")
print(Vt)

# 재구성
A_reconstructed = U @ Sigma @ Vt
print("Reconstruction error:")
print(np.linalg.norm(A - A_reconstructed))

# Rank-k 근사
k = 1
A_k = U[:, :k] @ Sigma[:k, :k] @ Vt[:k, :]
print("Rank-1 approximation:")
print(A_k)
```

### 🎨 시각화
```
SVD 의 기하학적 의미:

Input R²         V rotation       Σ scaling        U rotation        Output R²
     ┌─────┐           ┌─────┐             ┌─────┐               ┌─────┐
     │     │   rotate  │     │   scale     │     │   rotate      │     │
     └─────┘   --------> ───────>  --------> ───────>  --------> ───────>
     2x2                   2x2                    2x2                  2x2
```

### 🔧 실전 활용
- **Dimensionality reduction**: rank-k 근사
- **Image compression**: SVD 로 압축
- **Recommendation systems**: latent factors

### 📝 체크 문제
1. 특잇값과 고윳값의 관계는?
2. `rank(A)`는 SVD 에서 어떻게 읽는가?

---

## 9. 주성분 분석 (PCA)

### 🎯 핵심 질문
**"왜 PCA 가 차원 축소 표준 해답인가?"**

### 📐 직관 설명
**분산을 최대화하는 축**을 찾아 데이터의 핵심 방향 포착

```
A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ
```

### 💻 계산 실습

```python
import numpy as np
from sklearn.decomposition import PCA

# 예시 데이터
np.random.seed(0)
data = np.random.randn(100, 3)

# SVD 기반 PCA
U, s, Vt = np.linalg.svd(data, full_matrices=False)
# 스케일링 보정
s = s / np.sqrt(len(data) - 1)

# 주성분 분산 설명
explained_var = s**2
total_var = explained_var.sum()
print("Explained variance ratio:")
print(explained_var / total_var)

# scikit-learn 활용
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)
print("Explained variance:", pca.explained_variance_ratio_)
```

### 🎨 시각화
```
PCA 의 시각화:

Before PCA:          After PCA:
   y                    PC2
   ^                   ^
   |  ●                |      ● ●
   | ● ●              |   ●       ●
   |●    ●            | ●           ●
   +-----> x          +-----> PC1
   (3D)                (2D, 분산 최대)
```

### 🔧 실전 활용
- **Data compression**: 이미징, 영상 처리
- **Noise reduction**: noisy 데이터 클린징
- **Visualization**: 고차원 데이터를 2D/3D 로

### 📝 체크 문제
1. PCA 와 SVD 의 관계는?
2. 왜 1 번째 주성분이 분산을 최대화하는가?

---

## 10. 레일리 몫과 일반화 고유값

### 🎯 핵심 질문
**"왜 `Av = λMv` 꼴이 실전 문제에서 나오는가?"**

### 📐 직관 설명
`M`-가중 내적공간에서의 고유문제

```
R(x) = (xᵀAx) / (xᵀMx)
min R(x) 또는 max R(x)
```

### 💻 계산 실습

```python
import numpy as np

A = np.array([[4, 2],
              [2, 3]])
M = np.array([[2, 1],
              [1, 2]])  # SPD

# 일반화 고유값 문제
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M, A))

print("Generalized eigenvalues:", eigenvalues)

# 레일리 몫 계산
def rayleigh_ratio(x, A, M):
    return (x.T @ A @ x) / (x.T @ M @ x)

x = eigenvectors[:, 0]
print("Rayleigh quotient:", rayleigh_ratio(x, A, M))
print("Eigenvalue:", eigenvalues[0])  # 동일
```

### 🎨 시각화
```
일반화 고유값:

M=I 일 때:            M≠I 일 때:
   y                    M-내적으로 측정한 거리
   |                    등고선이 타원
   |   elliptical    |
   |     contours    |
   +---> x          +---> x
```

### 🔧 실전 활용
- **LDA (Linear Discriminant Analysis)**: 분류 문제
- **Vibration analysis**: 물리계 고유진동수
- **Constrained optimization**: 제약 조건 내 최적화

### 📝 체크 문제
1. LDA 에서 분모/분자가 각각 무엇을 의미하는가?
2. M 이 비가역일 때 어떻게 처리하는가?

---

## 11. 노름 (Norms)

### 🎯 핵심 질문
**"왜 노름을 바꾸면 최적해가 달라지는가?"**

### 📐 직관 설명
노름 = **거리의 정의**. 규칙이 바뀌면 최적해도 바뀜

| 노름 | 정의 | 기하학적 의미 |
|------|------|--------------|
| L1 | `||x||₁ = Σ|xᵢ|` | 마름모꼴 (희소 유도) |
| L2 | `||x||₂ = √Σxᵢ²` | 원형 (가장 일반적) |
| Linf | `||x||∞ = max|xᵢ|` | 정사각형 |

### 💻 계산 실습

```python
import numpy as np

x = np.array([3, 4])

l1 = np.sum(np.abs(x))           # 7
l2 = np.linalg.norm(x, 2)        # 5.0
linf = np.linalg.norm(x, np.inf)  # 4

print(f"L1: {l1}, L2: {l2}, L∞: {linf}")

# 희소성 유도 실습
from sklearn.linear_model import Lasso, Ridge

X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 0, 0, 3]) + 0.1 * np.random.randn(100)

lasso = Lasso(alpha=0.1).fit(X, y)
ridge = Ridge().fit(X, y)

print("Lasso (sparse):", lasso.coef_)
print("Ridge (dense):", ridge.coef_)
```

### 🎨 시각화
```
L1 vs L2 노름의 등고선:

L2 (원형)          L1 (마름모)
   _                    _
 _/ \_               _/   \_
|  O  |             |  ◊    |
 |_ _/              |       |
                    -+--
```

### 🔧 실전 활용
- **Regularization**: L1 (Lasso) vs L2 (Ridge)
- **Sparse coding**: 희소 표현 학습
- **Robust optimization**: 외란에 강인한 해

### 📝 체크 문제
1. L1 정규화가 희소해를 유도하는 기하학적 이유?
2. 행렬 노름 `||A||₂`는 무엇을 의미하는가?

---

## 12. 행렬/텐서 분해: NMF 와 텐서

### 🎯 핵심 질문
**"왜 SVD 외에 NMF/텐서 분해를 써야 하나?"**

### 📐 직관 설명
- **SVD**: 최적성 ✅, 해석성 ❌ (부호 섞임)
- **NMF**: 부분의 합으로 해석 ✅ (비음수)
- **Tensor**: 다차원 구조 보존 ✅

### 💻 계산 실습

```python
import numpy as np
from sklearn.decomposition import NMF

# 예시 데이터 (비음수)
np.random.seed(0)
X = np.random.rand(10, 5) * 10

# NMF 분해
nmf = NMF(n_components=3, random_state=0)
W = nmf.fit_transform(X)      # 10×3
H = nmf.components_           # 3×5

print("W (samples × components):")
print(W)
print("H (components × features):")
print(H)

# 재구성
X_reconstructed = W @ H
print("Reconstruction:")
print(X_reconstructed)

# 비음수 확인
print("All non-negative:", np.all(W >= 0) and np.all(H >= 0))
```

### 🎨 시각화
```
NMF 의 부분합 해석:

이미지 = 부분 1 + 부분 2 + 부분 3
      = (눈)      + (입)      + (머리)

SVC vs NMF:
SVC: 부호 섞여 해석 어려움
NMF: 각 부분이 의미 있는 요소로 해석 가능
```

### 🔧 실전 활용
- **Topic modeling**: 문서의 주제 분해
- **Face recognition**: 얼굴 부분 특징 추출
- **Music separation**: 악기별 신호 분리
- **Multiway data**: 텐서로 다차원 데이터 처리

### 📝 체크 문제
1. NMF 와 SVD 중 얼굴 특징 해석에 더 적합한 것은?
2. CP 와 Tucker 분해의 차이?

---

## 📊 추가 학습 자료

### 🧪 NumPy/PyTorch 실습 프로젝트
1. 행렬 분해 라이브러리 구현
2. PCA 로 이미지 압축
3. NMF 로 토픽 모델링
4. SVD 로 추천 시스템

### 📚 추천 읽을거리
- **기하학적 직관**: "Linear Algebra Done Right" - Axler
- **통계적 관점**: "Introduction to Linear Algebra" - Strang
- **ML 응용**: "Pattern Recognition and Machine Learning" - Bishop

---

*최종 업데이트: 2026-04-02*  
*직관적 이해와 실전 계산을 연결하는 보조 자료*
