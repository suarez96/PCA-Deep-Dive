# Weekend Project: Principal Component Analysis (PCA)

For full notebook interactivity and better formatted markdown: https://nbviewer.org/github/suarez96/PCA-Deep-Dive/blob/main/derivation_notebook.ipynb

The goal of principal component anaylsis (PCA) is to take a set of data in $D$-dimensional space, and compress it into as few dimensions as possible while still retaining the maximum amount of information. The reduced dimensions attempt to maximize retained information, measured as explained variance, through linear combinations of the original features. Successful applications of PCA allow us to greatly reduce the problem space, in particular for data with many superfluous dimensions where most of the variation can be explained through a few orthogonal linear combinations of features. This process is analogous to lossy compression, where if we use all the available dimensions, decompressing will fully recover the original data. Likewise, as we limit the size of the reduced dimensions, we attempt to capture almost the entire information with fewer data points, but are unable to fully recover the original data after decompression. 

It is important to highlight that we are not simply selecting some features with weight 1 and others with weight 0, but we are creating a matrix that will project our original data $x$ onto a $k$-dimensional subspace where hopefully our data's important characteristics remain unchanged. We call this matrix $U_a$ and it is defined as the $k$-leftmost columns of the matrix $U$, where $k$ is our number of selected principal components. $k$ will also be the resulting dimension of our data after 'compression'. More detail follows:

## Computation of PCA and Inverse PCA

For the block matrix
$$U = \begin{bmatrix}
\overbrace{U_{a}}^{\text{PC's}} & \overbrace{U_{b}}^{\text{excluded PC's}}
\end{bmatrix}
$$
We can create a projection $P$ of our original vector $x$ onto our new subspace through the operation

$$P=xU_a \tag{1.1}$$

Note that our transformation matrix $U \in \mathbb{R}^D$ is an orthonormal basis, and as such is unitary, ie. $UU^T=I_D$, the $D$-dimensional identity matrix. $U_a$ is not guaranteed to be unitary, but $U_aU_a^T$ should approach $I_D$ as more and more components are added. When all components are included, naturally, the equality is satisfied, since $U_a$ becomes $U$. The unitary property allows us to efficiently compute the inverse PCA transformation of our projected data $P$, through the operation:

$$\hat{x} = PU_a^T \tag{1.2}$$

Note that as $U_a\rightarrow U$, $\hat{x} = PU_a^T = xU_aU_a^T \rightarrow xUU^T = x$ (See figure below)

$\hat{x}$ is our reconstructed data point, which under a well-suited PCA fit, will be close to $x$. We can measure this 'reconstruction error' through the mean squared error (MSE) of our reconstructed data.

$$MSE=\frac{\sum_{i=0}^{N}(\hat{x}_{i}-x_{i})^2}{N}$$


## Mathematical Derivation
It is fairly common knowledge that PCA can be achieved through the eigendecomposition of the feature correlation matrix, but it is less commonly known that this explained-variance-maximization is equivalent to the minimization of the MSE. The proof is as follows:

Start with a vector $x$ and its reconstruction $\text{PCA}(x) = P \rightarrow \text{Inverse PCA}(P) = \hat{x}$. To minimize $MSE(\hat{x})$ we setup the following unconstrained optimization:

$$\text{minimize} \quad {\frac{\sum_{i=1}^{N}||\hat{x_i}-x_i||^2}{N}}$$
where $\hat{x_i}$ equals the projection of $x_i$ onto the unit vector $U_a$ , multiplied by $U_a$. ie:
$$\hat{x_i} = (x_i \cdot U_a) U_a$$
Squared error for a single sample, therefore becomes,

$$||\hat{x_i}-x_i||^2 = {||(x_i \cdot U_a) U_a - x_i||^2}$$
$$= ((x_i \cdot U_a) U_a - x_i)((x_i \cdot U_a) U_a - x_i) $$
$$= ((x_i \cdot U_a) U_a)^2 \underbrace{- x_i \cdot (x_i \cdot U_a) U_a - (x_i \cdot U_a) U_a \cdot x_i}_{\text{rearrange dot products} \rightarrow -2(x_i \cdot U_a)(x_i \cdot U_a)=-2(x_i \cdot U_a)^2}  + \underbrace{x_i\cdot x_i}_{=||x_i||^2} $$
$$= (x_i \cdot U_a)^2 \underbrace{U_a \cdot U_a}_{||U_a||^2=1^2=1} - 2(x_i \cdot U_a)^2 + ||x_i||^2$$
$$= (x_i \cdot U_a)^2 - 2(x_i \cdot U_a)^2 + ||x_i||^2$$
$$= ||x_i||^2 - (x_i \cdot U_a)^2$$

Over all terms, the mean squared error is then defined as

$$\frac{\sum_{i=1}^{N}||x_i||^2 - (x_i \cdot U_a)^2}{N} $$
$$ = \frac{\sum_{i=1}^{N}||x_i||^2}{N} - \frac{\sum_{i=1}^{N}(x_i \cdot U_a)^2}{N} \tag{2.1}$$

Note that the first term in eq $(2.1)$ is always going to be non-negative and is not going to depend on our choice of $U_a$, meaning that the problem 
$$\text{minimize} \quad {\frac{\sum_{i=1}^{N}||\hat{x_i}-x_i||^2}{N}}$$
is equivalent to maximizing the second term in eq. $(2.1)$.
$$\text{maximize} \quad \frac{\sum_{i=1}^{N}(x_i \cdot U_a)^2}{N} \tag{2.2}$$
Here we make note of the variance formula for a vector $v$. $Var(v)=E[v^2]-E[v]^2$. For our vector of projections, we have
$P = \textbf{x}U_a = \begin{bmatrix} 
x_1 \cdot U_a\\
x_2 \cdot U_a\\
... \\
x_N \cdot U_a\\
\end{bmatrix} $ where
$E[P] = \begin{bmatrix} 
E[x_1 \cdot U_a]\\
E[x_2 \cdot U_a]\\
... \\
E[x_N \cdot U_a]\\
\end{bmatrix}  = \begin{bmatrix} 
E[x_1] \cdot U_a\\
E[x_2] \cdot U_a\\
... \\
E[x_N] \cdot U_a\\
\end{bmatrix} = \begin{bmatrix} 
0\\
0\\
... \\
0\\
\end{bmatrix} $
our variance is exactly equal to $E[P^2]$, which is exactly what eq. $(2.2)$ is maximizing.

### Maximizing Variance/Minimizing MSE
For a single vector $u$:
$$\text{maximize} \quad \sigma^2=\frac{\sum_{i=1}^{N}(x_i \cdot u)^2}{N}$$
$$\textrm{s.t.} \quad u^Tu=1$$
Our constraint ensures $u$, our projection vector, is of unit length.
We start by restating our cost function as $$\sigma^2=\frac{\sum_{i=1}^{N}(x_i \cdot u_i)^2}{N} = \frac{(x_i u)^T(x_i u)}{N} = \frac{u^Tx^Tx u}{N} \tag{2.3}$$
We can make use of the fact that our data vector $x$ is zero mean and that for a vector $y$, its correlation matrix $R_y$ $$ = E[(y-\mu_y)(y-\mu_y)^T] = E[yy^T]$$
Taking $y=x^T$ 
$$R_{x^T} = E[x^Tx] = \frac{x^Tx}{N}$$Thus, eq $(2.3)$ simplifies to
$$u^TR_{x^T}u$$
To optimize, we write the lagrangian as follows:
$$\mathbb{L} = u^TR_{x^T}u  - \nu(uu^T-1)$$
where <em>nu</em>, $\nu$, is our lagrange multiplier for equality constraints. We differentiate with respect to $u$ and set to zero to find the optimum

$$\frac{\partial\mathbb{L}}{\partial{u}}=\frac{\partial{u^TR_{x^T}u}}{\partial{u}}-\frac{\partial{(\nu u^Tu-\nu)}}{\partial{u}}=\frac{2R_{x^T}u}{N}-2\nu u=0$$

$$\therefore R_{x^T}\textbf{u}=\nu \textbf{u}$$

We know that for a scalar $\lambda$, a matrix $A$ and vector $v$, if $A\textbf{v}=\lambda \textbf{v}$ then $v$ is an eigenvector of $A$ with corresponding eigenvalue $\lambda$. It is easy to see now that to maximize the variance in our optimization problem, we need to pick the eigenvector of $R_{x^T}$, $u$ with largest corresponding eigenvalue $\nu$. Q.E.D.

Helpful resources: 
- Boyd, Stephen, Stephen P. Boyd, and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.
- https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch18.pdf