# Weekend Project: Principal Component Analysis (PCA)

I am aware there are still a few inconsistencies in notation. The aim of this weekend project was to provide a straightforward example of a PCA use case, and to see how far we could get in a single weekend. Markdown has been cleaned up, but no code changes have been made since Sunday, March 5th.

UPDATE August 25th: [ChirPCA INTERACTIVE NOTEBOOK](https://nbviewer.org/github/suarez96/PCA-Deep-Dive/blob/main/ChirPCA/ChirPCA.ipynb) with much more interesting BirdCLEF data.

[INTERACTIVE DERIVATION NOTEBOOK](https://nbviewer.org/github/suarez96/PCA-Deep-Dive/blob/main/derivation_notebook.ipynb) (with better formatted markdown)

The goal of principal component anaylsis (PCA) is to take a set of data in $D$-dimensional space, and compress it into as few dimensions as possible while still retaining the maximum amount of information. The reduced dimensions attempt to maximize retained information, measured as explained variance, through linear combinations of the original features. Successful applications of PCA allow us to greatly reduce the problem space, in particular for data with many superfluous dimensions where most of the variation can be explained through a few orthogonal linear combinations of features. 

Through PCA we are creating a matrix that will project our original data $x$ onto a $k$-dimensional subspace where (ideally) our data's important characteristics still remain. Our projection matrix $\mathbf{U}_a$ will be defined by the $k$-leftmost columns of the matrix $\mathbf{U}$ (more detail below). $k$ is our number of selected principal components. This process is analogous to lossy compression, where the projection onto a lower-dimensional subspace will provide significant computational benefit or allow for simpler modeling, but we will be unable to fully recover the original data after decompression. The limiting case is that when we use all the available dimensions, decompressing will fully recover the original data, but no dimensioniality reduction will have been accomplished. 


## Computation of PCA and Inverse PCA

For now, it is only important to know that each column in $U$ is an eigenvector of the feature correlation matrix of $\mathbf{x}$, and that it is orthonormal. We justify this statement in the following section. 

For the block matrix
$$\mathbf{U} = \begin{bmatrix}
\overbrace{\mathbf{U}_{a}}^{\text{PC's}} & \overbrace{\mathbf{U}_{b}}^{\text{excluded PC's}}
\end{bmatrix}
$$
We can create a projection $\mathbf{p}$ of our original data vector $\mathbf{x} \in R^{N \times 1}$ onto our new subspace through the operation

$$ \mathbf{p}=\mathbf{x}^T \mathbf{U}_a \tag{1.1}$$

We posit that our transformation matrix, $U \in \mathbb{R}^D$, is an orthonormal basis, and therefore unitary, such that $\mathbf{U}\mathbf{U}^T=\mathbf{I}_D$ ($\mathbf{I}_D$ is the $D$-dimensional identity matrix). This property implies that by adding more and more PC's, $\mathbf{U}_a \rightarrow \mathbf{U}$ and $\mathbf{U}_a \mathbf{U}_a^T \rightarrow \mathbf{I}$. We will use this fact to compute the inverse PCA transformation of our projected data $\mathbf{p}$, through the operation:

$$\hat{\mathbf{x}} = \mathbf{p}\mathbf{U}_a^T \tag{1.2}$$ 

Under a well-suited PCA use case, $\hat{\mathbf{x}}$, our reconstructed data point, will be close to $\mathbf{x}$ even when we only use a small number of columns to build $\mathbf{U}_a$. We can measure this distance, known as the 'reconstruction error', through the mean squared error (MSE) to our original data vector $\mathbf{x}$.

$$MSE=\frac{\sum_{i=0}^{N}(\hat{x}_{i}-x_{i})^2}{N}$$


## Mathematical Derivation

PCA can be achieved through the eigendecomposition of the feature correlation matrix, and this explained-variance-maximization is equivalent to the minimization of the MSE. The proof is as follows:

Start with a vector $\mathbf{x}$ and its reconstruction $\text{PCA}(\mathbf{x}) = \mathbf{p} \rightarrow \text{Inverse PCA}(\mathbf{p}) = \hat{\mathbf{x}}$. To minimize $MSE(\hat{\mathbf{x}})$ we setup the following unconstrained optimization:

$$\text{minimize} \quad {\frac{\sum_{i=1}^{N}||\hat{x_i}-x_i||^2}{N}}$$
where $\hat{x_i}$ equals the projection of $x_i$ onto the unit vector $\mathbf{u} \in \mathbf{U}_a$ , multiplied by $\mathbf{u}$. ie:
$$\hat{x_i} = (x_i \cdot \mathbf{u}) \mathbf{u}$$
Squared error for a single sample, therefore becomes,

$$||\hat{x_i}-x_i||^2 = {||(x_i \cdot \mathbf{u}) \mathbf{u} - x_i||^2}$$
$$= ((x_i \cdot \mathbf{u}) \mathbf{U}_a - x_i)((x_i \cdot \mathbf{u}) \mathbf{u} - x_i) $$
$$= ((x_i \cdot \mathbf{u}) \mathbf{u})^2 \underbrace{- x_i \cdot (x_i \cdot \mathbf{U}_a) \mathbf{u} - (x_i \cdot \mathbf{u}) \mathbf{U}_a \cdot x_i}_{\text{rearrange dot products} \rightarrow -2(x_i \cdot \mathbf{u})(x_i \cdot \mathbf{u})=-2(x_i \cdot \mathbf{u})^2}  + \underbrace{x_i\cdot x_i}_{=||x_i||^2} $$
$$= (x_i \cdot \mathbf{u})^2 \underbrace{\mathbf{u} \cdot \mathbf{u}}_{||\mathbf{u}||^2=1^2=1} - 2(x_i \cdot \mathbf{u})^2 + ||x_i||^2$$
$$= (x_i \cdot \mathbf{u})^2 - 2(x_i \cdot \mathbf{u})^2 + ||x_i||^2$$
$$= ||x_i||^2 - (x_i \cdot \mathbf{u})^2$$

Over all terms, the mean squared error is then defined as

$$\frac{\sum_{i=1}^{N}||x_i||^2 - (x_i \cdot \mathbf{u})^2}{N} $$
$$ = \frac{\sum_{i=1}^{N}||x_i||^2}{N} - \frac{\sum_{i=1}^{N}(x_i \cdot \mathbf{u})^2}{N} \tag{2.1}$$

Note that the first term in eq $(2.1)$ is always going to be non-negative and is not going to depend on our choice of $\mathbf{U}_a$, meaning that the problem 
$$\text{minimize} \quad {\frac{\sum_{i=1}^{N}||\hat{x_i}-x_i||^2}{N}}$$
is equivalent to maximizing the second term in eq. $(2.1)$.
$$\text{maximize} \quad \frac{\sum_{i=1}^{N}(x_i \cdot \mathbf{u})^2}{N} \tag{2.2}$$
Here we make note of the variance formula for a vector $\mathbf{v}$. $Var(\mathbf{v})=E[\mathbf{v}^2]-E[\mathbf{v}]^2$. For our vector of projections, we have
$\mathbf{p} = \mathbf{x} \mathbf{U}_a = \begin{bmatrix} 
x_1 \cdot \mathbf{u}\\
x_2 \cdot \mathbf{u}\\
... \\
x_N \cdot \mathbf{u}\\
\end{bmatrix} $ where
$E[\mathbf{p}] = \begin{bmatrix} 
E[x_1 \cdot \mathbf{u}]\\
E[x_2 \cdot \mathbf{u}]\\
... \\
E[x_N \cdot \mathbf{u}]\\
\end{bmatrix}  = \begin{bmatrix} 
E[x_1] \cdot \mathbf{u}\\
E[x_2] \cdot \mathbf{u}\\
... \\
E[x_N] \cdot \mathbf{u}\\
\end{bmatrix} = \begin{bmatrix} 
0\\
0\\
... \\
0\\
\end{bmatrix} $
our variance is exactly equal to $E[\mathbf{p}^2]$, which is exactly what eq. $(2.2)$ is maximizing.

### Maximizing Variance/Minimizing MSE
For a single vector $\mathbf{u}$:
$$\text{maximize} \quad \sigma^2=\frac{\sum_{i=1}^{N}(x_i \cdot \mathbf{u})^2}{N}$$
$$\textrm{s.t.} \quad \mathbf{u}^T\mathbf{u}=1$$
Our constraint ensures $\mathbf{u}$, our projection vector, is of unit length.
We start by restating our cost function as $$\sigma^2=\frac{\sum_{i=1}^{N}(x_i \cdot u_i)^2}{N} = \frac{(x_i u)^T(x_i u)}{N} = \frac{u^Tx^Tx u}{N} \tag{2.3}$$
We can make use of the fact that our data vector $\mathbf{x}$ is zero mean and that for a vector $\mathbf{y}$, its correlation matrix $R_y$ $$ = E[(\mathbf{y}-\mu_y)(\mathbf{y}-\mu_y)^T] = E[\mathbf{y}\mathbf{y}^T]$$
Taking $\mathbf{y}=\mathbf{x}^T$ 
$$R_{\mathbf{x}^T} = E[\mathbf{x}^T\mathbf{x}] = \frac{\mathbf{x}^T\mathbf{x}}{N}$$Thus, eq $(2.3)$ simplifies to
$$\mathbf{u}^TR_{\mathbf{x}^T}\mathbf{u}$$
To optimize, we write the lagrangian as follows:
$$\mathbb{L} = \mathbf{u}^TR_{\mathbf{x}^T}\mathbf{u}  - \nu(\mathbf{u}\mathbf{u}^T-1)$$
where <em>nu</em>, $\nu$, is our lagrange multiplier for equality constraints. We differentiate with respect to $\mathbf{u}$ and set to zero to find the optimum

$$\frac{\partial\mathbb{L}}{\partial{\mathbf{u}}}=\frac{\partial{\mathbf{u}^TR_{\mathbf{x}^T}\mathbf{u}}}{\partial{\mathbf{u}}}-\frac{\partial{(\nu \mathbf{u}^T\mathbf{u}-\nu)}}{\partial{\mathbf{u}}}=2R_{\mathbf{x}^T}\mathbf{u}-2\nu \mathbf{u}=0$$

$$\therefore R_{\mathbf{x}^T}\mathbf{u}=\nu \mathbf{u}$$

We know that for a scalar $\lambda$, a matrix $A$ and vector $\mathbf{v}$, if $A\mathbf{v}=\lambda \mathbf{v}$ then $\mathbf{v}$ is an eigenvector of $A$ with corresponding eigenvalue $\lambda$. It is trivial to see that the eigenvector of $R_{\mathbf{x}^T}$, $\mathbf{u}$ with largest corresponding eigenvalue $\nu$, maximizes our optimization problem. Q.E.D.

Helpful resources: 
- Boyd, Stephen, Stephen P. Boyd, and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.
- https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch18.pdf