# Parallel Mean-Covariance Model

This repository implements a **parallel mean-covariance model** for population neural activity recorded from macaque dorsolateral prefrontal cortex (dlPFC) during a freely moving foraging task.

The project combines a conditional mean model with a low-rank, temporally smooth latent covariance model to investigate how behavioral, spatial, and strategic variables shape both firing rates and shared neural variability.

The dataset is based on the work *Population coding of strategic variables during foraging in freely moving macaques*, which demonstrated that strategic reward variables are represented in macaque dlPFC during naturalistic foraging behavior. The covariance modeling approach is additionally informed by *Investigating Inter-Area Covariance in the Primate Frontoparietal Reach Network via Latent Space Modelling*, an unpublished M.Sc. thesis by Rene Burghardt at the University of Göttingen.

---

## Model

For a trial with behavioral and spatial covariates \(x\), the neural response vector \(y\) is modeled as:

$$
y \mid x \sim \mathcal{N}\left(\mu_\theta(x), \Sigma_\phi\right)
$$

The conditional mean model predicts the expected activity of each unit and time bin:

$$
\hat{\mu}(x) = f_\theta(x)
$$

The covariance model captures structured residual variability after the conditional mean has been fitted. The covariance is represented as a low-rank latent process plus independent noise:

$$
\Sigma =
\sum_{k=1}^{K} \Lambda_k K_k \Lambda_k^{\top}
+
\operatorname{diag}(\sigma^2)
$$

where:

- \(\Lambda_k\) contains the loading pattern of latent component \(k\).
- \(K_k\) is a temporally smooth covariance kernel.
- \(\sigma^2\) represents unit- and time-specific residual variance.

The conditional mean model is trained first and then frozen while the covariance parameters are optimized. Therefore, the covariance model is intended to capture shared neural variability that is not already explained by the conditional mean.

The joint Gaussian formulation also supports conditional prediction. For target dimensions \(A\) and observed dimensions \(B\):

$$
\mathbb{E}[y_A \mid y_B]
=
\mu_A
+
\Sigma_{AB}\Sigma_{BB}^{-1}(y_B-\mu_B)
$$

The corresponding conditional covariance is:

$$
\Sigma_{A\mid B}
=
\Sigma_{AA}
-
\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}
$$

These equations allow neural activity to be predicted from previous activity, other units, or both.

---

## Dataset and Variables

Freely moving macaques performed a self-paced foraging task in which two operant buttons delivered probabilistic rewards while the animals navigated an enclosure.

Neural activity was recorded from dlPFC and binned at 200 ms around button presses. Each trial contains behavioral, strategic, spatial, and neural variables.

### Strategic variables

- `tslp` — Time since the previous press.
- `tunp` — Time until the next press.
- `rew_ratio` — Recent reward ratio associated with the current location.
- `rew` — Reward outcome of the current press.
- `choice` — Whether the animal switched to the other location after the press.
- `last_choice` — Whether the current press followed a recent switch.
- `rew_rate` — Reward-rate-related task variable used in the preprocessing pipeline.

### Spatial variables

- `x` — Horizontal position.
- `y` — Vertical position.
- `d` — Distance to the nearest reward source or reference point.
- `motion` — Movement or motion-energy estimate.

These variables are combined into trial-level and time-varying inputs for the conditional mean model.

---

## Preprocessing

The preprocessing pipeline includes:

- MATLAB-based extraction and filtering.
- Conversion of continuous time into 200-ms bins.
- Extraction of peri-press neural and behavioral windows.
- Removal of invalid, incomplete, or non-foraging trials.
- Rejection of unstable or contaminated units.
- Removal of trials with missing values or tracking failures.
- Robust spike outlier detection using median and median absolute deviation.
- Log transformation of strongly right-skewed interval variables.
- Standardization using training-set statistics only.
- Variance stabilization of spike counts using the Anscombe transform.
- An 80/20 train-validation split with fixed indices reused across model variants.

The preprocessing pipeline is designed to prevent information leakage between training and validation data.

---

## Prediction Regimes

Because the model defines a joint Gaussian distribution over units and time bins, it supports several conditional prediction regimes:

- **Mean:** Prediction from the conditional mean alone.
- **Past:** Conditioning on previous time bins of the same unit.
- **Others:** Conditioning on other units at the same time bin.
- **Past and others:** Joint temporal and population conditioning.

These regimes allow the contribution of temporal covariance and cross-unit covariance to be evaluated separately.

---

## Evaluation

Model performance is evaluated on held-out trials using:

- Pearson correlation coefficient.
- Coefficient of determination, \(R^2\).
- Mean-squared error.
- Negative log-likelihood.
- Unit-level comparisons between model variants.
- Paired statistical tests across validation trials.
- Benjamini–Hochberg false-discovery-rate correction where multiple comparisons are performed.

The main comparisons are:

1. Baseline versus conditional mean model.
2. Conditional mean versus mean-covariance model.
3. Mean-only prediction versus temporal conditioning.
4. Mean-only prediction versus cross-unit conditioning.
5. Within-group conditioning versus cross-group conditioning.

---

## SHAP Interpretability

SHAP values are used to attribute model-predicted firing to behavioral and spatial variables.

Because exact Shapley-value computation is expensive, the implementation estimates feature contributions using randomly sampled feature permutations. For each permutation, variables are added sequentially and the corresponding changes in predicted activity are accumulated.

The analysis includes:

- Single-trial SHAP explanations.
- Feature dependence plots.
- Population-level attribution maps.
- Feature-ranking plots.
- Shuffle-null comparisons.
- Null-standardized SHAP values.
- Variable-level selectivity.
- Functional-class selectivity.

Shuffle-null models destroy the trial-level relationship between neural activity and input variables while preserving their marginal distributions. This provides a reference for distinguishing meaningful attribution from attribution that can arise by chance or model flexibility.

SHAP values and covariance structure should be interpreted as model-derived explanations and statistical dependencies. They do not by themselves establish biological causality or anatomical connectivity.

---

## Selectivity Analysis

Variable-level selectivity is quantified from trial-averaged absolute null-standardized SHAP values across the peri-event window.

The analysis identifies units that selectively encode individual variables and then groups variables into broader functional classes:

- **Movement:** `x`, `y`, `d`, `motion`.
- **Reward prediction:** `tslp`, `last_choice`, `rew_ratio`.
- **Reward outcome:** `rew`.
- **Action planning:** `tunp`, `choice`.

Selectivity is declared only when both statistical significance and a minimum standardized effect-size criterion are satisfied.

This avoids labeling nearly every unit as selective solely because a very small effect becomes statistically significant in a large analysis.

---

## Cross-Group Covariance

Units can be grouped according to reward-prediction and action-planning selectivity:

- Prediction only.
- Planning only.
- Both prediction and planning.
- Neither.

The group-structured covariance analysis asks whether shared variability is organized by functional role rather than by unit identity or firing rate alone.

For an ordered source-target group pair, target activity is predicted from source-group residuals and compared with two baselines:

- **Over mean:** Source-group conditioning compared with the conditional mean alone.
- **Over within:** Source-group conditioning compared with conditioning on other units within the target group.

Prediction improvement is assessed across held-out trials using paired statistical tests with multiple-comparison correction.

This analysis tests predictive covariance between functional groups. It does not, by itself, establish directed communication or causal influence.

---

## Main Figure

<p align="center">
  <img src="1.jpg" alt="Trained mean-covariance model" width="900">
</p>

**Figure 1.** Combined visualization of the trained mean-covariance model. The figure summarizes the fitted conditional mean and the learned latent covariance structure. The mean-model results describe prediction of neural activity, whereas the covariance results show the learned latent loading patterns, temporal length scales, and independent noise structure. The figure is intended as a model characterization and validation summary, not as direct evidence of causal communication between neural populations.

---

## Repository Structure

```text
mean-cov-model/
├── mean-cov-model.ipynb       # Main analysis notebook
├── filter_data.m              # MATLAB data filtering and preparation
├── data.mat                   # Prepared data file
├── src/                       # Model and analysis modules
├── 1.jpg                      # Combined model figure
└── LICENSE
```

Repository links:

- [`mean-cov-model.ipynb`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/mean-cov-model.ipynb)
- [`filter_data.m`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/filter_data.m)
- [`data.mat`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/data.mat)
- [`src/`](https://github.com/ZareiShayan/mean-cov-model/tree/real-data/src)
- [`LICENSE`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/LICENSE)

---

## References

- Shahidi, N., Franch, M., Parajuli, A., Schrater, P., Wright, A., Pitkow, X., & Dragoi, V. (2024). *Population coding of strategic variables during foraging in freely moving macaques*. **Nature Neuroscience, 27**, 772–781. https://doi.org/10.1038/s41593-024-01575-w

- Burghardt, R. *Investigating Inter-Area Covariance in the Primate Frontoparietal Reach Network via Latent Space Modelling*. Unpublished M.Sc. thesis, University of Göttingen.

---

## Citation

If you use this repository, please cite the dataset paper and methodological reference listed above and link to this repository.

---

## License

See [`LICENSE`](https://github.com/ZareiShayan/mean-cov-model/blob/real-data/LICENSE).
