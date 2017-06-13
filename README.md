# pyRMT
Python for Random Matrix Theory. Implements several cleaning schemes for noisy correlation matrices, 
including the optimal shrinkage, rotationally-invariant estimator
to an underlying correlation matrix (as proposed by Joel Bun, 
Jean-Philippe Bouchaud, Marc Potters and colleagues).

Such cleaned correlation matrix are known to improve factor-decomposition
via Principal Component Analysis (PCA) and could be of relevance in a variety 
of contexts, including computational biology.

Cleaning schemes also result in much improved out-of-sample risk
of Markowitz optimal portfolios, as established over the years
in several papers by Jean-Philippe Bouchaud, Marc Potters and collaborators.

Some cleaning schemes can be easily adapted from the various shrinkage
estimators implemented in the sklearn.covariance module 
(see the various publications by O. Ledoit and M. Wolf listed below).

In addition, it might make sense to perform an empirical estimate
of a correlation matrix robust to outliers before proceeding with
the cleaning schemes of the present module. Some of those robust estimates
have been implemented in the sklearn.covariance module as well. 


References
----------
* "DISTRIBUTION OF EIGENVALUES FOR SOME SETS OF RANDOM MATRICES",
  V. A. Marcenko and L. A. Pastur, Mathematics of the USSR-Sbornik, Vol. 1 (4), pp 457-483
* "A well-conditioned estimator for large-dimensional covariance matrices",
  O. Ledoit and M. Wolf, Journal of Multivariate Analysis, Vol. 88 (2), pp 365-411
* "Improved estimation of the covariance matrix of stock returns with "
  "an application to portfolio selection",
  O. Ledoit and M. Wolf, Journal of Empirical Finance, Vol. 10 (5), pp 603-621
* "Financial Applications of Random Matrix Theory: a short review",
  J.-P. Bouchaud and M. Potters, arXiv: 0910.1205 [q-fin.ST]
* "Eigenvectors of some large sample covariance matrix ensembles",
  O. Ledoit and S. Peche, Probability Theory and Related Fields, Vol. 151 (1), pp 233-264
* "NONLINEAR SHRINKAGE ESTIMATION OF LARGE-DIMENSIONAL COVARIANCE MATRICES",
  O. Ledoit and M. Wolf, The Annals of Statistics, Vol. 40 (2), pp 1024-1060 
* "Rotational invariant estimator for general noisy matrices",
  J. Bun, R. Allez, J.-P. Bouchaud and M. Potters, arXiv: 1502.06736 [cond-mat.stat-mech]
* "Cleaning large Correlation Matrices: tools from Random Matrix Theory",
  J. Bun, J.-P. Bouchaud and M. Potters, arXiv: 1610.08104 [cond-mat.stat-mech]
  

Attribution
-----------

If you happen to use pyRMT in your work or research, please cite its GitHub repository:

G. Giecold, pyRMT, (2017), GitHub repository, https://github.com/GGiecold/pyRMT

The respective BibTex entry is

```
@misc{GregoryGiecold2017,
  author = {G. Giecold},
  title = {pyRMT},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/GGiecold/pyRMT}}
}
```


License
-------

Copyright 2017-2022 Gregory Giecold and contributors.

pyRMT is free software made available under the MIT License. For details see the LICENSE file.
