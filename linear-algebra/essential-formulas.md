# Essential Linear Algebra Formulas

A comprehensive collection of must-know linear algebra formulas for mathematics, physics, engineering, and data science applications.

---

## 1. Matrix Operations

### Basic Operations

**Matrix Addition/Subtraction:**
$$ (A + B)_{ij} = A_{ij} + B_{ij} $$

**Matrix Multiplication:**
$$ (AB)_{ij} = \sum_{k=1}^n A_{ik}B_{kj} $$

**Transpose:**
$$ (A^T)_{ij} = A_{ji} $$
Properties: $(A^T)^T = A$, $(A+B)^T = A^T + B^T$, $(AB)^T = B^T A^T$

**Inverse:**
$$ A^{-1}A = AA^{-1} = I $$
Properties: $(A^{-1})^{-1} = A$, $(AB)^{-1} = B^{-1}A^{-1}$, $(A^T)^{-1} = (A^{-1})^T$

---

## 2. Determinants

**2×2 Matrix:**
$$ \det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc $$

**3×3 Matrix (Sarrus' Rule):**
$$ \det\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = aei + bfg + cdh - ceg - bdi - afh $$

**Properties:**
- $\det(A^T) = \det(A)$
- $\det(AB) = \det(A)\det(B)$
- $\det(A^{-1}) = \frac{1}{\det(A)}$
- $\det(cA) = c^n\det(A)$ for n×n matrix

**Cofactor Expansion:**
$$ \det(A) = \sum_{j=1}^n (-1)^{i+j}A_{ij}\det(M_{ij}) $$
where $M_{ij}$ is the minor matrix.

**Cramer's Rule:**
For $Ax = b$:
$$ x_i = \frac{\det(A_i)}{\det(A)} $$
where $A_i$ is A with column i replaced by b.

---

## 3. Vector Operations

### Inner Products

**Dot Product:**
$$ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta $$

**Cross Product (3D only):**
$$ \mathbf{a} \times \mathbf{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix} = (a_2b_3 - a_3b_2, a_3b_1 - a_1b_3, a_1b_2 - a_2b_1) $$

### Vector Spaces

**Norms:**
- L1 norm: $\|\mathbf{x}\|_1 = \sum |x_i|$
- L2 norm (Euclidean): $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$
- L∞ norm: $\|\mathbf{x}\|_\infty = \max_i |x_i|$

**Distance:**
$$ d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum (x_i - y_i)^2} $$

**Projection of u onto v:**
$$ \text{proj}_{\mathbf{v}}\mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\mathbf{v} $$

---

## 4. Eigenvalues and Eigenvectors

**Definition:**
$$ A\mathbf{v} = \lambda\mathbf{v} $$
where $\lambda$ is eigenvalue, $\mathbf{v}$ is eigenvector (non-zero).

**Characteristic Equation:**
$$ \det(A - \lambda I) = 0 $$

**Properties:**
- Sum of eigenvalues = trace(A)
- Product of eigenvalues = det(A)
- Real symmetric matrices have real eigenvalues
- Orthogonal eigenvectors for distinct eigenvalues (symmetric A)

**Diagonalization:**
$$ A = PDP^{-1} $$
where P has eigenvectors as columns, D is diagonal with eigenvalues.

---

## 5. Decompositions

### LU Decomposition
$$ A = LU $$
where L is lower triangular, U is upper triangular. Used for solving linear systems.

### QR Decomposition
$$ A = QR $$
where Q is orthogonal ($Q^TQ = I$), R is upper triangular. Used for least squares.

### Cholesky Decomposition (for symmetric positive-definite matrices)
$$ A = LL^T $$
where L is lower triangular.

### Singular Value Decomposition (SVD)
$$ A = U\Sigma V^T $$
- U: m×m orthogonal matrix (left singular vectors)
- Σ: m×n diagonal matrix with singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
- V: n×n orthogonal matrix (right singular vectors)

**Singular values relate to eigenvalues:**
$$ \sigma_i = \sqrt{\lambda_i(A^TA)} $$

### Spectral Decomposition (for symmetric matrices)
$$ A = Q\Lambda Q^T $$
where Q contains orthonormal eigenvectors, Λ is diagonal eigenvalue matrix.

---

## 6. Linear Systems

**Gaussian Elimination:**
Transform augmented matrix [A|b] to row echelon form.

**Back Substitution:**
For upper triangular system Ux = y, solve from bottom up.

**Least Squares Solution:**
Minimize $\|Ax - b\|^2$:
$$ A^TAx = A^Tb $$
$$ x = (A^TA)^{-1}A^Tb = A^\dagger b $$
where $A^\dagger = (A^TA)^{-1}A^T$ is the pseudoinverse for full column rank.

**Normal Equations:**
The equations $A^TAx = A^Tb$ derived from setting gradient to zero.

---

## 7. Vector Calculus

### Gradient
$$ \nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} $$

### Jacobian Matrix
For f: ℝⁿ → ℝᵐ:
$$ J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} $$

### Divergence
$$ \nabla \cdot \mathbf{F} = \frac{\partial F_1}{\partial x_1} + \frac{\partial F_2}{\partial x_2} + \frac{\partial F_3}{\partial x_3} $$

### Curl (3D)
$$ \nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ F_1 & F_2 & F_3 \end{vmatrix} $$

### Hessian Matrix (2nd derivatives)
$$ H_f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix} $$

---

## 8. Subspaces and Bases

### Fundamental Subspaces
- **Column space (Range):** Col(A) = {Ax : x ∈ ℝⁿ}
- **Null space (Kernel):** Null(A) = {x : Ax = 0}
- **Row space:** Row(A) = Col(A^T)
- **Left null space:** Null(A^T)

**Dimension Theorem (Rank-Nullity):**
$$ \text{rank}(A) + \text{nullity}(A) = n $$

### Basis Properties
- Linearly independent spanning set
- All bases of same subspace have same cardinality
- Change of basis matrix P: $[x]_B = P[x]_C$

---

## 9. Orthogonality

### Orthogonal Matrix
$$ Q^TQ = QQ^T = I $$
- Columns form orthonormal basis
- Preserves lengths: $\|Qx\| = \|x\|$
- Preserves dot products: $(Qx) \cdot (Qy) = x \cdot y$
- $\det(Q) = \pm 1$

### Orthonormal Basis
- Vectors are orthogonal and unit length
- Gram matrix: $Q^TQ = I$
- Coordinate transformation: $P_{xy} = Q^T x$

### Gram-Schmidt Process
Given basis {v₁, v₂, ..., vₙ}:
$$ u_1 = v_1 $$
$$ u_k = v_k - \sum_{j=1}^{k-1} \frac{v_k \cdot u_j}{\|u_j\|^2}u_j $$
$$ e_k = \frac{u_k}{\|u_k\|} $$

---

## 10. Special Matrix Types

### Symmetric Matrix
$$ A = A^T $$
- All eigenvalues are real
- Orthogonally diagonalizable

### Skew-Symmetric Matrix
$$ A = -A^T $$
- Diagonal elements are 0
- Eigenvalues are purely imaginary or zero

### Orthogonal Matrix
$$ A^TA = I $$
- Columns are orthonormal
- Inverse equals transpose

### Projection Matrix
$$ P = P^2 = P^T $$
Projection onto column space of A:
$$ P = A(A^TA)^{-1}A^T $$

### Rotation Matrix (2D)
$$ R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} $$

### Rotation Matrix (3D - about axis)
$$ R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix} $$
(And similarly for y and z axes)

---

## 11. Tensor Operations

### Outer Product
$$ \mathbf{u} \otimes \mathbf{v} = \mathbf{u}\mathbf{v}^T $$

### Kronecker Product
$$ A \otimes B = \begin{bmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{bmatrix} $$

### Trace
$$ \text{tr}(A) = \sum_{i=1}^n A_{ii} $$
Properties: tr(A) = tr(A^T), tr(AB) = tr(BA), tr(A+B) = tr(A) + tr(B)

---

## 12. Optimization Basics

### Gradient Descent
$$ x_{k+1} = x_k - \alpha\nabla f(x_k) $$

### Taylor Expansion (Multivariate)
$$ f(x+h) \approx f(x) + \nabla f(x)^T h + \frac{1}{2}h^T H_f(x)h $$

### Lagrange Multipliers
Maximize $f(x)$ subject to $g(x) = 0$:
$$ \nabla f(x) = \lambda \nabla g(x) $$

---

## 13. Important Identities

**Matrix Inverse (Sherman-Morrison):**
$$ (A + uv^T)^{-1} = A^{-1} - \frac{A^{-1}uv^TA^{-1}}{1 + v^TA^{-1}u} $$

**Matrix Determinant Lemma:**
$$ \det(A + uv^T) = \det(A)(1 + v^TA^{-1}u) $$

**Woodbury Matrix Identity:**
$$ (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1} $$

**Cauchy-Schwarz Inequality:**
$$ |\mathbf{a} \cdot \mathbf{b}| \leq \|\mathbf{a}\|\|\mathbf{b}\| $$

**Triangle Inequality:**
$$ \|\mathbf{a} + \mathbf{b}\| \leq \|\mathbf{a}\| + \|\mathbf{b}\| $$

**Pseudoinverse Properties:**
- $(A^\dagger)^\dagger = A$
- $(A^T)^\dagger = (A^\dagger)^T$
- $(AB)^\dagger = B^\dagger A^\dagger$ (under certain conditions)

---

## Quick Reference: Common Eigenvalues

- Identity: All λ = 1
- Zero matrix: All λ = 0
- Diagonal matrix: λ = diagonal elements
- Projection matrix: λ ∈ {0, 1}
- Orthogonal matrix: |λ| = 1
- Symmetric positive-definite: All λ > 0
- Skew-symmetric: λ is purely imaginary or zero

---

## References

- Strang, Gilbert. "Linear Algebra and Its Applications"
- Lay, David C. "Linear Algebra and Its Applications"
- Axler, Sheldon. "Linear Algebra Done Right"
- Boyd, Stephen. "Convex Optimization"

---

*Last updated: March 2026*
