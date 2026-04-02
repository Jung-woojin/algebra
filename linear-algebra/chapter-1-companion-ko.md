# Chapter 1 보조자료: 선형대수 직관, 엄밀성, 실전

이 문서는 행렬과 벡터의 **기하학적 의미**, **수학적 엄밀성**, **실제 계산**을 모두 다루는 종합 자료입니다.  
이론과 실무를 연결하는 "다리" 역할을 합니다.

---

## 📚 사용 가이드

### 학습 방법
1. **핵심 질문** → 개념의 본질 이해
2. **수학적 엄밀성** → 엄밀한 정의와 정리
3. **증명의 핵심** → 논리적 흐름 파악
4. **직관 설명** → 머릿속에 그림 그리기
5. **계산 예제** → 손으로 직접 풀어보기
6. **시각화** → 그래프로 확인하기
7. **실전 활용** → 실제 문제에 적용

### 이 문서의 특징
- ✅ **기하학 중심**: 행렬을 "변환"으로 이해
- ✅ **수학적 엄밀**: 정리·정의·증명 포함
- ✅ **실전 계산**: 이론과 실제 계산 연결
- ✅ **시각화 지원**: 그림으로 직관화
- ✅ **코드 예제**: NumPy/PyTorch 실습

---

## 1. 행렬-벡터 곱셈 Ax: 열의 선형결합

### 🎯 핵심 질문
**"Ax 를 왜 행렬-벡터 곱이 아니라 '열의 조합'으로 봐야 하나?"**

### 📐 수학적 엄밀성

#### [정리 1.1] 행렬 - 벡터 곱의 열 표현
> **Theorem**: 임의의 m × n 행렬 `A`와 n 차원 벡터 `x`에 대해  
> `A = [a₁ a₂ ... aₙ]`, `x = [x₁ x₂ ... xₙ]ᵀ`이면  
> **Ax = ∑ᵢ₌₁ⁿ xᵢaᵢ**

**증명 (핵심 아이디어)**:
```
A x = [a₁ a₂ ... aₙ] [x₁]
                    [x₂]
                    ...
                    [xₙ]
    
    = x₁a₁ + x₂a₂ + ... + xₙaₙ  (행렬 곱셈 정의에 의해)
```

**수학적 의미**:
- `Ax`는 `A`의 열벡터들의 **선형결합**
- `x`는 각 열에 대한 **계수 (coefficient)**
- `Col(A) = {Ax | x ∈ ℝⁿ}` (열공간 정의)

### 💡 직관 설명
`A`의 각 열에 `x`의 성분이 가중치로 작용:
```
Ax = x₁ × (열 1) + x₂ × (열 2) + ... + xₙ × (열 n)
```

### 📝 계산 예시
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

# 방법 2: 열의 선형결합 (엄밀한 계산)
col1 = A[:, 0]  # array([1, 0])
col2 = A[:, 1]  # array([2, 1])
Ax2 = 3 * col1 + 4 * col2  # array([11, 4])

print("Verification: A @ x = sum(x_i * a_i)")
print(f"A @ x = {Ax1}, sum = {Ax2}, equal: {np.allclose(Ax1, Ax2)}")
```

### 🎨 시각화
```
기하학적 의미:
x = [3, 4]² → x 가 basis 벡터들의 가중치
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

A 의 열: a₁=(1,0), a₂=(2,1)
x=(3,4) → Ax = 3a₁ + 4a₂ = (11, 4)
```

### 🔧 실전 활용
- **Computer Graphics**: 변환 행렬로 객체 회전/이동
- **Neural Networks**: fully connected layer = Ax
- **Signal Processing**: 필터 적용 = Convolution

### 📝 체크 문제
1. `rank(A) < n`이면 어떤 벡터 `x ≠ 0`에 대해 `Ax = 0`이 되는가?  
   **해**: `x ∈ N(A)` (영공간에 속하는 임의의 벡터)
2. `b`가 `Col(A)`에 있지 않으면 `Ax = b`는 어떤 상태인가?  
   **해**: 해가 존재하지 않음 (부정계)

---

## 2. 행렬 곱셈 AB: 변환의 합성

### 🎯 핵심 질문
**"AB 의 기하학적 의미는 무엇인가?"**

### 📐 수학적 엄밀성

#### [정리 2.1] 행렬 곱셈의 선형변환 합성
> **Theorem**: `A`가 m × n 행렬, `B`가 n × p 행렬일 때,  
> 선형변환 `T_A(x) = Ax`, `T_B(x) = Bx`에 대해  
> **T_A ∘ T_B = T_{AB}** 즉, `(AB)x = A(Bx)`

**증명**:
```
(AB)x = A(Bx)  (행렬 곱셈의 결합법칙)
      = T_A(T_B(x))
```

**수학적 의미**:
- `AB`는 `B` 이후 `A`를 적용하는 **선형변환의 합성**
- `B`가 먼저 변환 → `A`가 다시 변환
- `(AB)_{ij}`는 `A`의 i 행과 `B`의 j 열의 **내적**

### 💡 직관 설명
```
AB = [Ab₁ Ab₂ ... Abₙ]
     = A × (B 의 열 1), A × (B 의 열 2), ...
```

### 📝 계산 예시
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
# B: x 축 2 배, y 축 그대로
# A: x 축 그대로, y 축 2 배
# AB: x 축 2 배, y 축 4 배
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
   **해**: `Col(AB) ⊆ Col(A)`이므로 차원이 작을 수 없음
2. `AB = 0`이지만 `A ≠ 0`, `B ≠ 0`일 수 있는가?
   **해**: 예, `Col(B) ⊆ N(A)`인 경우 (예: `A=[1,0]`, `B=[0;1]`)

---

## 3. 네 가지 기본 부분공간

### 🎯 핵심 질문
**"왜 열공간, 행공간, 영공간, 좌영공간 4 개를 함께 봐야 하나?"**

### 📐 수학적 엄밀성

#### [정리 3.1] 기본 부분공간의 차원 관계
> **Theorem**: `A`가 m × n 행렬, `rank(A) = r`일 때:
> - `dim(Row(A)) = dim(Col(A)) = r`
> - `dim(N(A)) = n - r` (nullity)
> - `dim(N(Aᵀ)) = m - r`

**증명의 핵심**:
1. **Rank-Nullity Theorem**: `dim(Col(A)) + dim(N(A)) = n`
2. **Row space ≅ Col(Aᵀ)**: 행공간과 열공간의 차원은 동일

#### [정리 3.2] 기본 부분공간의 직교 관계
> **Theorem**:  
> `Row(A) ⟂ N(A)`  
> `Col(A) ⟂ N(Aᵀ)`

**수학적 의미**:
- 입력공간: `ℝⁿ = Row(A) ⊕ N(A)` (직교합)
- 출력공간: `ℝᵐ = Col(A) ⊕ N(Aᵀ)` (직교합)

### 💡 직관 설명
입력과 출력 공간에서 **"살아남는 방향"**과 **"사라지는 방향"**:
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
- **System 의 해 존재성**: `b ∈ Col(A)` 여부 확인
- **과소/과대 결정**: `rank`, `n`, `m` 관계 확인
- **Stability 분석**: condition number 계산

### 📝 체크 문제
1. `A`가 5×3, rank=2 일 때 `N(A)`의 차원은?  
   **해**: `n - r = 3 - 2 = 1`
2. `Ax=b`가 해를 갖기 위한 필요충분조건은?  
   **해**: `b ∈ Col(A)`

---

## 4. 소거법과 A = LU 분해

### 🎯 핵심 질문
**"왜 LU 분해가 실전 계산의 핵심인가?"**

### 📐 수학적 엄밀성

#### [정리 4.1] LU 분해의 존재성
> **Theorem**: `A`가 n × n 정사각행렬이고 모든 leading principal minor 가 non-zero 일 때,  
> **A = LU** (P=I 인 경우) 가 유일하게 존재한다.  
> 여기서 `L`은 하삼각, `U`는 상삼각이다.

**소거법과 LU 의 관계**:
```
소거: Eₖ ... E₂E₁A = U
L = E₁⁻¹E₂⁻¹ ... Eₖ⁻¹  →  A = LU
```

**수학적 의미**:
- 소거계수를 모으면 `L`, 상삼각화한 결과를 `U`
- `A = LU`로 저장하면 `Ax=b` 를 `Ly=b`, `Ux=y` 로 빠르게 풀 수 있음

### 💡 직관 설명
```
A = L × U
    ↑   ↑
   하   상
  삼각  삼각
```

### 📝 계산 예시
```
A = [2  1  1]
    [4  3  3]
    [8  7  9]

L = [1  0  0]    U = [2  1  1]
    [2  1  0]        [0 -1 -1]
    [4  3  1]        [0  0  2]
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
   **해**: 소거가 불가능하므로, row swap 으로 피벗 조정 (`PA = LU`)
2. `A = LU`가 항상 존재하는가?  
   **해**: 아니요, leading principal minor 가 0 인 경우 존재하지 않음

---

## 5. 직교행렬과 부분공간

### 🎯 핵심 질문
**"왜 직교행렬이 수치적으로 중요한가?"**

### 📐 수학적 엄밀성

#### [정의 5.1] 직교행렬
> **Definition**: 정사각행렬 `Q` 가 **직교행렬 (orthogonal matrix)**이란  
> **QᵀQ = I** 또는 동치인 **Q⁻¹ = Qᵀ**를 만족하는 것이다.

#### [정리 5.1] 직교행렬의 성질
> **Theorem**: `Q` 가 직교행렬일 때:
> 1. 열벡터들은 **서로 직교하며 단위벡터** (orthonormal)
> 2. **||Qx|| = ||x||** (길이 보존)
> 3. **<Qx, Qy> = <x, y>** (내적 보존)
> 4. **|det(Q)| = 1**

**수학적 의미**:
- `Q`는 **회전 또는 반사** (isometry)
- 수치적으로 **매우 안정적** (condition number = 1)

### 💡 직관 설명
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
   **해**: 서로 직교하며 단위벡터 (orthonormal)
2. `Q`가 직교행렬일 때 `det(Q)`의 가능한 값은?  
   **해**: `±1` (회전이면 `1`, 반사이면 `-1`)

---

## 6. 고윳값과 고유벡터

### 🎯 핵심 질문
**"고유벡터를 왜 찾는가?"**

### 📐 수학적 엄밀성

#### [정의 6.1] 고유값과 고유벡터
> **Definition**: 정사각행렬 `A` 에 대해,  
> 스칼라 `λ` 와 영벡터가 아닌 벡터 `v`가 **`Av = λv`**를 만족하면,  
> - `λ` 를 **고유값 (eigenvalue)**,  
> - `v` 를 **고유벡터 (eigenvector)**라고 한다.

#### [정리 6.1] 고유값의 대수적 성질
> **Theorem**:
> 1. `λ` 가 고유값 ⇔ **det(A - λI) = 0** (특성방정식)
> 2. **대각화 가능**: `A = PDP⁻¹` (고유벡터가 기저를 이룰 때)
> 3. **닮음행렬**: `B = P⁻¹AP` 는 `A` 와 같은 고유값

#### [정리 6.2] 대칭행렬의 고유값 정리
> **Theorem (Spectral Theorem)**: 실수 대칭행렬 `A` 에 대해:
> - 모든 고유값은 **실수**
> - 서로 다른 고유값에 대응하는 고유벡터는 **서로 직교**

### 💡 직관 설명
고유벡터: **방향은 유지되고 크기만 바뀜**

```
Av = λv
```

### 📝 계산 예시
```
A = [4 1]      Av = [4 1][1] = [5] = 5[1]
    [2 3]             [2 3][1]   [5]      [1]

고유값 λ = 5, 고유벡터 v = [1, 1]ᵀ
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
   **해**: `n` 개의 선형독립 고유벡터가 존재할 때 (즉, `rank(A - λI) = n - geometric multiplicity`)
2. `A`와 `P⁻¹AP` 는 왜 같은 고유값을 가지는가?  
   **해**: `det(P⁻¹AP - λI) = det(P⁻¹(A - λI)P) = det(A - λI)`

---

## 7. 대칭 양의 정부호 (SPD) 행렬

### 🎯 핵심 질문
**"왜 SPD 가 최적화/ML 에서 중요한가?"**

### 📐 수학적 엄밀성

#### [정의 7.1] SPD 행렬
> **Definition**: 정사각행렬 `S` 가 다음을 만족하면 **SPD (Symmetric Positive Definite)**:
> 1. **대칭**: `S = Sᵀ`
> 2. **양의 정부호**: **∀x ≠ 0, xᵀSx > 0**

#### [정리 7.1] SPD 의 동치 조건
> **Theorem**: 다음 명제들은 모두 동치:
> 1. `S` 는 SPD
> 2. 모든 고유값이 **양수**
> 3. 모든 **주축소行列式 (principal minors)**이 양수 (Sylvester's criterion)
> 4. **Cholesky 분해** `S = LLᵀ` 가능 (`L`은 하삼각)
> 5. **xᵀSx**는 `ℝⁿ` 위에서 **강한 볼록함수**

#### [정리 7.2] AᵀA 의 SPD 성질
> **Theorem**: 임의의 실수행렬 `A` 에 대해 **AᵀA**는 PSD (Positive Semi-Definite)  
> **A**의 rank 가 `n` (full column rank) 일 때, **AᵀA**는 SPD

**수학적 의미**:
- 모든 방향에서 양의 에너지 = **볼록한 그릇**
- 최적화에서 **단일 전역최소점** 보장

### 💡 직관 설명
모든 방향에서 양의 에너지 = **볼록한 그릇**

```
xᵀSx > 0  (∀ x ≠ 0)
```

### 📝 계산 예시
```
A = [4 2]      AᵀA = [5  2]  (SPD)
    [2 3]              [2 13]

고유값: λ₁ ≈ 1.5, λ₂ ≈ 15.5 (모두 양수)
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
   **해**: `xᵀ(AᵀA)x = ||Ax||² ≥ 0` (노름의 비음수성)
2. SPD 판별 3 가지 방법?  
   **해**: 고윳값이 모두 양수, 모든 주축소行列식 > 0, Cholesky 분해 가능

---

## 8. 특잇값 분해 (SVD)

### 🎯 핵심 질문
**"왜 모든 행렬이 SVD 로 분해 가능한가?"**

### 📐 수학적 엄밀성

#### [정리 8.1] SVD 의 존재성과 유일성
> **Theorem** (Fundamental Theorem of SVD):  
> 임의의 m × n 행렬 `A` 에 대해,  
> **A = UΣVᵀ** 로 분해 가능하다.  
> 여기서:
> - `U` (m × m) 와 `V` (n × n) 는 직교행렬
> - `Σ` (m × n) 는 대각성분만 가지는 사각행렬 (특잇값 σ₁ ≥ σ₂ ≥ ... ≥ 0)
> - **특잇값은 유일**하며, **σᵢ = √λᵢ(AᵀA)**

#### [정리 8.2] SVD 의 기하학적 의미
> **Theorem**: SVD `A = UΣVᵀ` 는 다음 3 단계 변환:
> 1. `Vᵀ`: 입력공간에서 직교기저로 회전
> 2. `Σ`: 축 방향으로 스케일링 (σᵢ 배)
> 3. `U`: 출력공간에서 직교기저로 회전

#### [정리 8.3] 에카트 - 영 (Eckart-Young) 정리
> **Theorem**: `A_k = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ`는 **rank k** 행렬 중  
> `||A - A_k||_F` 를 최소화하는 최적 근사이다.

**수학적 의미**:
- 임의의 선형변환을 **직교 + 스케일 + 직교**로 분해
- `rank(A) = number of non-zero σᵢ`

### 💡 직관 설명
임의의 선형변환을 **3 단계 회전/스케일**로 분해

```
A = U Σ Vᵀ

V: 입력 축 회전
Σ: 축별 스케일링
U: 출력 축 회전
```

### 📝 계산 예시
```
A = [3 0]      U = [0  1]    Σ = [5  0]    Vᵀ = [0  1]
    [4 5]            [1  0]        [0  5]            [1  0]

A = UΣVᵀ: 직교 회전 → 스케일 → 직교 회전
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
   **해**: σᵢ = √λᵢ(AᵀA) (AᵀA 의 고윳값의 제곱근)
2. `rank(A)`는 SVD 에서 어떻게 읽는가?  
   **해**: non-zero 특잇값의 개수

---

## 9. 주성분 분석 (PCA)

### 🎯 핵심 질문
**"왜 PCA 가 차원 축소 표준 해답인가?"**

### 📐 수학적 엄밀성

#### [정리 9.1] PCA 와 SVD 의 동치성
> **Theorem**: 데이터 행렬 `X` (n × d, 표준화됨) 에 대해:
> - 공분산 행렬: `C = XᵀX / (n-1)`
> - PCA 는 `C` 의 **고유값 분해**
> - **PCA = SVD(X)**와 수학적으로 동치

#### [정리 9.2] 최적 저랭크 근사
> **Theorem** (에카트 - 영):  
> `A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ`는 **모든 rank k 행렬 중**  
> `||A - A_k||_F²` 를 최소화한다.  
> 최소 오차: **∑ᵢ₌ₖ₊₁ʳ σᵢ²**

**수학적 의미**:
- **분산 최대화** = **재구성 오차 최소화** (같은 문제의 다른 표현)
- 주성분 = 공분산 행렬의 고유벡터 (고유값 순)

### 💡 직관 설명
**분산을 최대화하는 축**을 찾아 데이터의 핵심 방향 포착

```
A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ
```

### 📝 계산 예시
```
데이터: 3 차원 벡터들
PCA 2 차원으로 축소:
PC1: 분산 최대 축
PC2: PC1 에 직교하며 나머지 분산 최대
손실: PC3 방향의 정보
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
   **해**: PCA(SVD) = `SVD(X)`에서 `V`를 주성분으로 사용
2. 왜 1 번째 주성분이 분산을 최대화하는가?  
   **해**: 고윳값이 최대인 고유벡터 방향으로의 투영이 분산 최대

---

## 10. 레일리 몫과 일반화 고유값

### 🎯 핵심 질문
**"왜 `Av = λMv` 꼴이 실전 문제에서 나오는가?"**

### 📐 수학적 엄밀성

#### [정의 10.1] 일반화 고유값 문제
> **Definition**: `A`와 `M`(SPD) 에 대해,  
> 스칼라 `λ` 와 벡터 `v ≠ 0`이 **`Av = λMv`**를 만족할 때,  
> - `λ` 를 **일반화 고유값**,  
> - `v` 를 **일반화 고유벡터**라고 한다.

#### [정리 10.1] 일반화 고유값의 성질
> **Theorem**: `M` 이 SPD 일 때:
> 1. 모든 고유값은 **실수**
> 2. **M-직교성**: `vᵢᵀMvⱼ = 0` (i ≠ j)
> 3. **변환**: `B = M⁻¹/²AM⁻¹/²`는 대칭, 일반화 고유값 = `B`의 고유값

#### [정리 10.2] 레일리 몫의 성질
> **Definition**: **레이일리 몫** `R(x) = (xᵀAx)/(xᵀMx)`  
> **Theorem**:
> - **최대값** = 최대 고유값, 최적 `x` = 최대 고유벡터
> - **최소값** = 최소 고유값, 최적 `x` = 최소 고유벡터

**수학적 의미**:
- `M`-가중 내적공간에서의 고유문제
- `M`이 유클리드 내적이 아닐 때의 일반화

### 💡 직관 설명
`M`-가중 내적공간에서의 고유문제

```
R(x) = (xᵀAx) / (xᵀMx)
min R(x) 또는 max R(x)
```

### 📝 계산 예시
```
A = [4 2]      M = [2 1]
    [2 3]          [1 2]

일반화 고유값: Av = λMv
λ₁ ≈ 1.12, λ₂ ≈ 7.88
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
   **해**: 분모 = within-class scatter, 분자 = between-class scatter
2. M 이 비가역일 때 어떻게 처리하는가?  
   **해**: pseudoinverse 사용, 또는正则化 추가

---

## 11. 노름 (Norms)

### 🎯 핵심 질문
**"왜 노름을 바꾸면 최적해가 달라지는가?"**

### 📐 수학적 엄밀성

#### [정의 11.1] 노름 (Norm)
> **Definition**: 함수 `||·||` 가 다음 3 조건을 만족하면 **노름**:
> 1. **양의 정부호**: `||x|| ≥ 0`, 등호는 `x = 0`일 때만
> 2. **동차성**: `||αx|| = |α|·||x||`
> 3. **삼각부등식**: `||x + y|| ≤ ||x|| + ||y||`

#### [정리 11.1] 주요 노름과 성질
> **Theorem**:
> - **L1 노름**: `||x||₁ = Σ|xᵢ|` → **희소성 유도**
> - **L2 노름**: `||x||₂ = √Σxᵢ²` → **유클리드 거리**
> - **L∞ 노름**: `||x||∞ = max|xᵢ|` → **최대 성분**
> - **행렬 노름**: `||A||₂ = σ_max(A)` (스펙트럴 노름)
> - **프로베니우스 노름**: `||A||_F = √Σaᵢⱼ²`

**수학적 의미**:
- 노름은 **거리의 정의**
- 최적화 문제에서 **정규화 항**으로 사용
- L1: 희소성, L2: 매끄러움

### 💡 직관 설명
노름 = **거리의 정의**. 규칙이 바뀌면 최적해도 바뀜

| 노름 | 정의 | 기하학적 의미 |
|------|------|--------------|
| L1 | `||x||₁ = Σ|xᵢ|` | 마름모꼴 (희소 유도) |
| L2 | `||x||₂ = √Σxᵢ²` | 원형 (가장 일반적) |
| Linf | `||x||∞ = max|xᵢ|` | 정사각형 |

### 📝 계산 예시
```
x = [3, 4]

||x||₁ = |3| + |4| = 7
||x||₂ = √(3² + 4²) = 5
||x||∞ = max(3, 4) = 4
```

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
   **해**: 마름모꼴 등고선이 축과 접할 때 (일부 계수 0)
2. 행렬 노름 `||A||₂`는 무엇을 의미하는가?  
   **해**: 최대 확대배율 (최대 특잇값)

---

## 12. 행렬/텐서 분해: NMF 와 텐서

### 🎯 핵심 질문
**"왜 SVD 외에 NMF/텐서 분해를 써야 하나?"**

### 📐 수학적 엄밀성

#### [정리 12.1] NMF (Non-negative Matrix Factorization)
> **Definition**: `X ≥ 0` (비음수 행렬) 에 대해,  
> **X ≈ UV** (U ≥ 0, V ≥ 0) 로 분해하는 문제.
>
> **Theorem**: NMF 는 **해석 가능성**이 우수하며,  
> **부분의 합 (parts-based representation)**을 제공한다.

#### [정리 12.2] 텐서 분해 (Tensor Decomposition)
> **CP 분해**: `T ≈ Σᵣ₌₁ᴿ aᵣ ⊗ bᵣ ⊗ cᵣ`  
> **Tucker 분해**: `T ≈ G ×₁ A ×₂ B ×₃ C`
>
> **Theorem**: 텐서 분해는 **다차원 구조 보존**에 우수

**수학적 의미**:
- **SVD**: 최적성 ✅, 해석성 ❌ (부호 섞임)
- **NMF**: 부분의 합으로 해석 ✅ (비음수)
- **Tensor**: 다차원 구조 보존 ✅

### 💡 직관 설명
- **SVD**: 최적성 ✅, 해석성 ❌ (부호 섞임)
- **NMF**: 부분의 합으로 해석 ✅ (비음수)
- **Tensor**: 다차원 구조 보존 ✅

### 📝 계산 예시
```
이미지 = 부분 1 + 부분 2 + 부분 3
      = (눈)      + (입)      + (머리)

SVC vs NMF:
SVC: 부호 섞여 해석 어려움
NMF: 각 부분이 의미 있는 요소로 해석 가능
```

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
   **해**: NMF (비음수 제약으로 부분 특징 추출 가능)
2. CP 와 Tucker 분해의 차이?  
   **해**: CP 는 직교합 (rank 1 텐서 합), Tucker 은 core tensor 로 유연한 분해

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
- **엄밀한 증명**: "Linear Algebra" - Hoffman & Kunze

---

*최종 업데이트: 2026-04-02*  
*수학적 엄밀성과 직관적 이해를 균형 있게 다룬 종합 보조 자료*
