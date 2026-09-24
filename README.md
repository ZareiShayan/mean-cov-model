# Parallel Mean-Covariance Model

This project studies neural population coding in freely moving macaque dorsolateral prefrontal cortex (dlPFC) during a self-paced foraging task, using data consistent with the "Population coding of strategic variables during foraging in freely moving macaques" dataset. Single-unit activity was recorded while animals pressed one of two operant buttons for probabilistic rewards, and spikes were binned at 200 ms around each press. The present work develops a latent mean-covariance neural model and interprets it with permutation-based SHAP attribution and null-standardized selectivity analysis.

---

## Model

The proposed framework is a **latent mean-covariance model** that jointly predicts a conditional mean firing-rate trajectory and a structured, low-rank population covariance. For a trial, the model outputs a mean vector $\mu(x)$ and a covariance factor $L$, defining a trial-wise multivariate Gaussian over the flattened bin-unit response vector $y$:

$$y \mid x \sim \mathcal{N}(\mu(x), \Sigma), \qquad \Sigma = LL^\top$$

The mean model is a transformer encoder that maps concatenated spatial and task covariates into a conditional mean trajectory $\mu(t, u \mid x)$ for each unit $u$ and time bin $t$:

$$\mu(t, u \mid x) = f_\theta\big(t2v(t),\; W_{vars} x_t\big)$$

where $x_t$ stacks position, dense-task, and sparse-task variables at bin $t$, $W_{vars}$ projects covariates into the hidden dimension, and $t2v(t)$ is a Time2Vec positional encoding combining linear and sinusoidal components of time.

The covariance model introduces shared latent factors that capture correlated variability across units and time. Each latent component $k$ has a loading matrix $\Lambda_k$ mapping into the flattened bin-unit space, with a temporally smooth Gaussian-process kernel controlling how loadings evolve across the trial:

$$K_k(t, t') = \exp\!\left(-\frac{(t-t')^2}{2\,\ell_k^2}\right)$$

where $\ell_k$ is the learned time scale of latent component $k$. The full covariance combines all latent components with a diagonal noise term:

$$\Sigma = \sum_k \Lambda_k \Lambda_k^\top + \mathrm{diag}(\sigma^2)$$

Latent loadings are initialized near zero so the model starts close to an identity covariance and gradually learns structure, while the mean model is trained first and then frozen so that the covariance component learns residual structure not already captured by the conditional mean.

---

## Interpretability

Model interpretability was assessed with **permutation-based SHAP** (SHapley Additive exPlanations). SHAP values allocate each unit's predicted firing-rate change at a given bin and trial to individual behavioral and spatial features by averaging marginal contributions over feature subsets:

$$\phi_i(t,u) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\Big[f(S \cup \{i\}) - f(S)\Big]$$

Because exact Shapley computation is infeasible, the implementation approximates $\phi_i$ via Monte Carlo permutations, and attributions are null-standardized against a shuffle-null ensemble to distinguish genuine variable-outcome relationships from baseline variance.

---

## Figure

<p align="center">
  <img src="assets/1.png" alt="Mean model training and covariance latent structure" width="800">
</p>

**Figure 1.** Trained mean and covariance model summary.
- **Top — Mean model training and evaluation.** Training and validation negative log-likelihood (NLL) across epochs, relative loss change, and the learning-rate schedule (left), alongside per-unit correlation, $R^2$, and MSE distributions for training and validation trials (right).
- **Bottom — Covariance latent structure before and after training.** Randomly initialized parameters (upper row) versus fitted values (lower row): latent covariance loadings across time bins and units (left), temporal length scales for the latent components (middle), and unit- and time-specific independent noise variances (right). The shared covariance model represents population covariance as a low-rank, temporally smooth latent process plus diagonal noise; loading magnitudes and signs determine each unit's contribution to each latent component, length scales govern how rapidly latent covariance can change across time, and the noise term captures residual variance not explained by shared latent structure.

---

## Research Questions

1. Does the covariance component provide predictive information about held-out firing rate beyond what the task-variable-dependent mean model already explains?
2. Are behaviorally relevant units selective for specific strategic variables (movement, reward prediction, reward outcome, action planning), and does this selectivity exceed a shuffle-null baseline?
3. Is population covariance organized by functional role — e.g., reward-prediction versus action-planning units — rather than by unit identity or anatomy alone?
4. Does conditioning on one functional group's activity improve prediction of another group's held-out responses beyond what within-group covariance already explains?

---

## Repository Structure

```
mean-cov-model/
├── mean-cov-model.ipynb   # End-to-end notebook: data → mean/covariance models → interpretation
├── filter_data.m          # MATLAB preprocessing and trial/unit filtering
├── data.mat               # Preprocessed spikes, position, and task-variable data
├── LICENSE
└── src/
    ├── classes.py          # Model and dataset class definitions
    ├── helper_functions.py # Preprocessing, training, and evaluation utilities
    └── plot_functions.py   # Visualization and figure-generation functions
```

Key source files:
- [`src/`](https://github.com/ZareiShayan/mean-cov-model/tree/real-data/src)
- [`src/plot_functions.py`](https://github.com/ZareiShayan/mean-cov-model/commit/77e0533b2fa27cdbd0aa2a01b36e103b4f43e283)
- [`filter_data.m`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/filter_data.m)
- [`data.mat`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/data.mat)
- [`mean-cov-model.ipynb`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/mean-cov-model.ipynb)
- [`LICENSE`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/LICENSE)

---

## Data and References

This project uses data and modeling approaches from the following sources:

- **Population coding of strategic variables during foraging in freely moving macaques** — the primary single-unit dlPFC foraging dataset used in this project.

- **Investigating Inter-Area Covariance in the Primate Frontoparietal Reach Network via Latent Space Modelling**, Rene Burghardt, MSc Thesis, University of Göttingen (unpublished) — the latent-space covariance modeling framework this project's covariance component builds on.

---

## Citation

If you use this code, please cite the two sources above and link to this repository.

---

## License

See [LICENSE](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/LICENSE).
