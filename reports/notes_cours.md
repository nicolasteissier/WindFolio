# Financial Big Data Project - Course Summary Notes

## Correlation Matrix C Definition:

$$C = \frac{1}{T} X'X = V'\Lambda V$$

where:
- $X = \text{data matrix of size } T \times N$
- $T = \text{number of timestamps}$

- $V = \text{matrix of eigenvectors of size } N \times N$
- $\Lambda = \text{diagonal matrix of eigenvalues of size } N \times N$

## Correlation Matrix Cleaning with RMT (Week 9):

### RMT Bounds

$$q = \frac{N}{T}$$
where:
- $N = \text{number of assets}$
- $T = \text{number of timestamps}$

If C = I (identity matrix) => no correlations between variables, then:

$$\forall \lambda, \lambda \in [(1-\sqrt{q})^2, (1+\sqrt{q})^2]$$

### Empirical Eigenvalues Clipping

To clean the correlation matrix from noise, we clip the eigenvalues outside the RMT bounds:

$$\lambda_i^{clip} = \delta, \text{ if  } \lambda_i \lt (1+\sqrt{q})^2$$


$$\lambda_i^{clip} = \lambda_i, \text{ if  } \lambda_i \geq (1+\sqrt{q})^2$$

where $\delta$ is the average of the eigenvalues that are below the upper RMT bound.

$$C^{(clip)} = V' \text{diag}(\Lambda^{clip}) V$$

Don't forget to set:
$$ C_{ii}^{clip} = 1 $$

---

## Correlation Matrix Cleaning with Rotationally Invariant Estimator (RIE) (Week 10):

### Optimal RIE for constant C:

$$\xi^{RIE}(\lambda) = \frac{1}{|1 - q + q \lambda \lim_{\nu \to 0^+} g_C(\lambda - i\nu)|^2}$$

where: idk shit

---

## Cross-Validation Approach (CV) for Correlation Matrix Cleaning (Week 10):

### CV Algorithm Steps:

1. Split the data into K random fractions of times.
2. For each fraction k:
   - Compute the correlation matrix $C^{out,k}$ using all data except fraction k.
   - Compute the eigenvectors $V^{in, k, T}$ of $C^{in,k}$.
   - Compute the oracle:
     $$\Lambda_{CV}^{(k)} = \text{diag}(V^{in, k, T} C^{out,k} V^{in, k})$$
3. Average over K folds:
   $$\Lambda_{CV} = \frac{1}{K} \sum_{k=1}^{K} \Lambda_{CV}^{(k)}$$

Note: Same exists for non-stationary data --> previous time period as in-sample and next time period as out-of-sample (cf. s24, Week 10 slides).

---

## ==> If we want to all of this:

- PyRMT package: https://pypi.org/project/pyrmt/
- nonlinshrinkage package: https://pypi.org/project/nonlinshrinkage/

--> NLS beats RIE and CV in terms of performance ==> use it (nonlinshrinkage package)

--- 

## Hierarchical filtering (Week 10):

Idea: We could use it to look for clusters in the wind data (e.g., regions with similar wind patterns), or different types of wind turbines ?

![Alt text](figures/hierarchical_filtering.png)

Package to use: BAHC: https://pypi.org/project/bahc/

Bonne chance.

---

## Noise Reduction:

### Dimensionality Reduction (clustering, ...)

We could use clustering techniques (e.g., K-means, Louvain) to group data from close meteorological stations. This would reduce the dimensionality of the dataset and help in identifying representative stations for each cluster, thereby reducing noise.

--> Use networkx + python-louvain for Louvain clustering.
--> Use scikit-learn for K-means clustering.










