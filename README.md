# Parallel Mean-Covariance Model

This project develops a **latent mean-covariance encoding model** for population spiking activity recorded from macaque dorsolateral prefrontal cortex (dlPFC) during a self-paced foraging task, and interprets the fitted model with permutation-based SHAP attribution. The task and neural data follow the paradigm and dataset introduced in *"Population coding of strategic variables during foraging in freely moving macaques."* The covariance component builds on the latent-space modeling approach developed in *"Investigating Inter-Area Covariance in the Primate Frontoparietal Reach Network via Latent Space Modelling"* (Rene Burghardt, MSc Thesis, University of Göttingen, unpublished).

---

## Task and Data

Freely moving macaques performed a self-paced foraging task with two operant buttons delivering probabilistic rewards. For each qualifying button press, single-unit dlPFC activity and animal position were extracted in a window from $$t=-3$$ s to $$t=+2$$ s relative to the press, binned at 200 ms, and later cropped to $$[-2,+2]$$ s for modeling. Each trial is described by a 7-dimensional strategic-variable vector (time since/until press, reward ratio, reward outcome, choice, and last choice) and a time-varying position tensor (x, y, distance to reward, motion).

---

## Model

The framework separates population activity into a **conditional mean** and a **latent, temporally structured covariance**, both computed from behavioral and spatial covariates.

**Baseline model.** A per-bin, per-unit template with identity covariance:

$$\mu_{t,u} = \theta_{t,u}$$

**Mean model.** A transformer encoder maps covariates $$x_t$$ into a conditional mean trajectory $$\mu(t,u \mid x)$$, with covariance still fixed to identity:

$$h_t = W_{vars}\, x_t + t2v(t)$$

**Covariance model.** Once the mean model is fit and frozen, a low-rank latent covariance is learned on top of it. Each latent component $$k$$ has a loading matrix $$\Lambda_k$$ and a Gaussian-process temporal kernel:

$$K_k(t,t') = \exp\!\left(-\frac{(t-t')^2}{2\ell_k^2}\right)$$

The full covariance combines all latent components with a diagonal noise term:

$$\Sigma = LL^\top, \qquad L \Rightarrow \Sigma = \sum_k \Lambda_k \Lambda_k^\top + \operatorname{diag}(\sigma^2)$$

**Joint mean-covariance likelihood.** The model outputs $$\mu(x)$$ and $$L$$, defining a trial-wise Gaussian over the flattened bin-unit response vector $$y$$:

$$-\log \mathcal{N}(y \mid \mu, \Sigma) = \tfrac{1}{2}(y-\mu)^\top \Sigma^{-1} (y-\mu) + \tfrac{1}{2}\log|\Sigma| + \tfrac{D}{2}\log(2\pi)$$

The mean model is trained first and frozen; the covariance parameters are optimized second, so the latent covariance captures residual structure not already explained by the conditional mean.

**Conditional prediction.** Because the model defines a joint Gaussian, it supports conditioning on subsets of dimensions (past bins, other units):

$$\mu_{A\mid B} = \mu_A + \Sigma_{AB}\Sigma_{BB}^{-1}(y_B-\mu_B), \qquad \Sigma_{A\mid B} = \Sigma_{AA} - \Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}$$

---

## Interpretability: Permutation SHAP

Model interpretability is assessed with **permutation-based Shapley attribution**, allocating each unit's predicted firing-rate change to individual behavioral and spatial features:

$$\phi_i(t,u) = \frac{1}{|N|!}\sum_{\pi} \left[ f(S_k \cup \{i\}) - f(S_k) \right]$$

Raw SHAP magnitudes are standardized against a shuffle-null ensemble to control for feature variance and unit noise, yielding a null-standardized attribution score in units of standard deviations above chance:

$$z_{s,i,t,u} = \frac{\bar{s}_{perm} - \bar{s}_{shuffle}}{\sqrt{(\sigma^2_{perm}+\sigma^2_{shuffle})/2}}$$

Variable selectivity is declared significant only when both an FDR-corrected permutation-versus-shuffle test ($$q<0.05$$) and a minimum standardized-effect-size threshold are satisfied, and significant variables are aggregated into four functional classes: movement, reward prediction, reward outcome, and action planning.

---

## Figure

<p align="center">
  <img src="assets/1.png" alt="Mean and covariance model training and structure" width="800">
</p>

**Figure 1.** Training and structure of the fitted mean-covariance model. *Top —* optimization trajectory of the conditional mean model, showing training/validation negative log-likelihood, relative loss change, learning-rate schedule, and the resulting distributions of per-unit correlation, $$R^2$$, and MSE. *Bottom —* evolution of the shared covariance-model parameters from random initialization (upper row) to fitted values (lower row): latent covariance loadings across time bins and units (left), temporal length scales for each latent component (middle), and unit- and time-specific independent noise variances (right). Together, these panels show the mean-covariance model learning a low-rank, temporally smooth latent covariance structure on top of the frozen conditional mean.

---

## Repository Structure

```
mean-cov-model/
├── mean-cov-model.ipynb   # End-to-end notebook: data → mean model → covariance model → SHAP
├── filter_data.m          # MATLAB preprocessing and trial/unit filtering
├── data.mat               # Preprocessed spikes, position, and task-variable data
├── LICENSE
└── src/
    ├── classes.py          # Mean, covariance, and joint model class definitions
    ├── helper_functions.py # Preprocessing, training, SHAP, and selectivity utilities
    └── plot_functions.py   # Visualization functions for figures above
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

- **Foraging dataset** — Data and task design follow *"Population coding of strategic variables during foraging in freely moving macaques."*
- **Covariance modeling approach** — The low-rank, temporally smooth latent covariance formulation builds on Burghardt, R. *"Investigating Inter-Area Covariance in the Primate Frontoparietal Reach Network via Latent Space Modelling."* MSc Thesis, University of Göttingen (unpublished).

---

## Citation

If you use this code, please cite the foraging dataset paper and the latent-space covariance modeling thesis above, and link to this repository.

---

## License

MIT
