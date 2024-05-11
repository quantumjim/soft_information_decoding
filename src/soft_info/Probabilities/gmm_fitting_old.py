# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from sklearn.mixture import GaussianMixture

# import warnings

# from ..Plotting.general import apply_formatting




# def estimate_gaussian_covariances_custom(resp, X, nk, means, reg_covar):
#     """
#     Custom covariance calculation for a GMM with 3 components, where the first two components share a tied (full)
#     covariance and the third component has its own full covariance.
#     """
#     n_components, n_features = means.shape
#     # assert n_components == 3, "The number of components must be 3 for custom covariance calculation."

#     covariances = np.empty((n_components, n_features, n_features))

#     if n_components == 3:
#         # Calculate tied covariance for the first two components
#         # Combined responsibilities for the first two components
#         resp_combined = resp[:, :2].sum(axis=1)
#         nk_combined = nk[:2].sum()

#         # Mean of the first two components weighted by their responsibilities
#         mean_combined = np.dot(resp_combined, X) / nk_combined

#         # Calculate the covariance matrix for the first two components
#         diff_combined = X - mean_combined
#         weighted_diff_combined = resp_combined[:, np.newaxis] * diff_combined
#         covariance_tied = np.dot(weighted_diff_combined.T, diff_combined) / nk_combined
#         covariance_tied.flat[::n_features + 1] += reg_covar  # Regularization

#         # Assign the tied covariance matrix to the first two components
#         covariances[0] = covariance_tied
#         covariances[1] = covariance_tied

#         # Calculate full covariance for the third component
#         diff = X - means[2]
#         weighted_diff = resp[:, 2][:, np.newaxis] * diff
#         covariances[2] = np.dot(weighted_diff.T, diff) / nk[2]
#         covariances[2].flat[::n_features + 1] += reg_covar

#     else: 
#         covariances = np.empty((n_components, n_features, n_features))
#         for k in range(n_components):
#             diff = X - means[k]
#             covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
#             covariances[k].flat[:: n_features + 1] += reg_covar

#     return covariances





# # def estimate_gaussian_covariances_custom(resp, X, nk, means, reg_covar):
# #     """Custom covariance calculation for a GMM with 3 components, where the first two components share a tied diagonal 
# #     covariance and the third component has its own full covariance.

# #     Usage
# #     -----
# #     To use this function for covariance estimation in sklearn's GMM:
# #     ```python
# #     from sklearn.mixture import _gaussian_mixture
# #     _gaussian_mixture._estimate_gaussian_covariances_full = _estimate_gaussian_covariances_custom
    
# #     gmm = _gaussian_mixture.GaussianMixture(n_components=3, covariance_type='full')
# #     gmm.fit(X)
# #     ```
# #     """

# #     n_components, n_features = means.shape
# #     if n_components == 3:
# #         # warnings.warn("Using custom covariance calculation for GMM with 3 components (tied diagonal and full covariance)")
# #         assert n_components == 3, "The number of components must be 3 for custom covariance calculation."

# #         covariances = np.zeros((n_components, n_features, n_features))

# #         # Calculate tied diagonal covariance for the first two components
# #         # Combined responsibilities for the first two components
# #         resp_combined = resp[:, :2].sum(axis=1)[:, np.newaxis]
# #         nk_combined = nk[:2].sum()  # Combined nk for the first two components

# #         avg_X2_combined = np.dot(resp_combined.T, X * X) / nk_combined
# #         avg_means2_combined = np.sum(means[:2]**2, axis=0, keepdims=True) / 2
# #         avg_X_means_combined = np.dot(
# #             resp_combined.T, X) * np.mean(means[:2], axis=0) / nk_combined

# #         covariance_tied_diag = avg_X2_combined - 2 * \
# #             avg_X_means_combined + avg_means2_combined + reg_covar
# #         covariances[0][:, :] = np.diag(covariance_tied_diag.ravel())
# #         covariances[1][:, :] = np.diag(covariance_tied_diag.ravel())

# #         # Calculate full covariance for the third component
# #         diff = X - means[2]
# #         covariances[2] = np.dot(resp[:, 2] * diff.T, diff) / nk[2]
# #         covariances[2].flat[::n_features + 1] += reg_covar
# #     else: 
# #         covariances = np.empty((n_components, n_features, n_features))
# #         for k in range(n_components):
# #             diff = X - means[k]
# #             covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
# #             covariances[k].flat[:: n_features + 1] += reg_covar

# #     return covariances





# def fit_gmm_0_calib(IQ_data):
#     gmm_0 = GaussianMixture(n_components=1, covariance_type='diag')
#     gmm_0.fit(process_complex_data(IQ_data))
#     return gmm_0


# def fit_gmm_1_calib(IQ_data, gmm_0, n_components):
#     assert gmm_0.covariance_type == 'diag', 'GMM_0 must have diagonal covariance'
#     assert n_components in [1, 2], 'n_components must be 1 or 2'

#     if n_components == 1:
#         precisions_init = 1 / gmm_0.covariances_
#         mean = np.array([[-gmm_0.means_[0, 0], gmm_0.means_[0, 1]]])

#         gmm_1 = GaussianMixture(n_components=n_components, covariance_type='diag',
#                                 means_init=mean, precisions_init=precisions_init)
#     elif n_components == 2:
#         mean1 = np.array([-gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
#         mean2 = np.array([-1.5 * gmm_0.means_[0, 0], 1.5 * gmm_0.means_[0, 1]])
#         means_init = np.array([mean1, mean2])
#         weights_init = np.array([0.9, 0.1])
#         precisions_init = np.array(
#             [np.diag(1 / gmm_0.covariances_[0]), np.diag(1 / gmm_0.covariances_[0])])

#         gmm_1 = GaussianMixture(n_components=n_components, covariance_type='full',
#                                 means_init=means_init, precisions_init=precisions_init,
#                                 weights_init=weights_init)

#     gmm_1.fit(process_complex_data(IQ_data))
#     return gmm_1



# # def fit_RepCode_data(IQ_data, gmm_0, gmm_1, n_components=[2, 3]):
# #     assert n_components in [
# #         2, 3], 'n_components not in [2, 3] is not supported currently'

# #     if n_components == 3:
# #         mean0 = np.array([gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
# #         mean1 = np.array([-gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
# #         mean2 = np.array([-2*gmm_0.means_[0, 0], 2*gmm_0.means_[0, 1]])
# #         mean2 = np.array(
# #             [0, gmm_0.means_[0, 1] + 0.5*np.abs(gmm_0.means_[0, 1])])
# #         means_init = np.array([mean0, mean1, mean2])

# #         gmm_RepCode = GaussianMixture(n_components=n_components, covariance_type='full',
# #                                       means_init=means_init, init_params="k-means++")

# #     elif n_components == 2:
# #         mean0 = np.array([gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
# #         mean1 = np.array([-gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
# #         means_init = np.array([mean0, mean1])

# #         gmm_RepCode = GaussianMixture(n_components=n_components, covariance_type='full',
# #                                       means_init=means_init, init_params="k-means++")


# #     gmm_RepCode.fit(process_complex_data(IQ_data))
# #     return gmm_RepCode



# def fit_RepCode_data(IQ_data, gmm_0, gmm_1, n_components=[2, 3], where='above'):
#     assert n_components in [
#         2, 3], 'n_components not in [2, 3] is not supported currently'

#     # Heuristic parameters (should be good)
#     tol = 1e-8
#     max_iter = 1_000

#     if n_components == 3:
#         mean0 = np.array([gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
#         mean1 = np.array([-gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
#         mean2 = np.array([0, gmm_0.means_[0, 1] + 0.5*np.abs(gmm_0.means_[0, 1])]) if where == 'above' else np.array(
#             [0, gmm_0.means_[0, 1] - 0.5*np.abs(gmm_0.means_[0, 1])])
#         means_init = np.array([mean0, mean1, mean2])
#         precisions_init = np.array([np.diag(1 / gmm_0.covariances_[0]), 
#                                     np.diag(1 / gmm_0.covariances_[0]), 
#                                     np.diag(1 / gmm_0.covariances_[0])])

#         gmm_RepCode = GaussianMixture(n_components=n_components, covariance_type='full',
#                                       means_init=means_init, precisions_init=precisions_init, init_params="k-means++", random_state=42,
#                                       tol=tol, max_iter=max_iter, n_init=1)

#     elif n_components == 2:
#         mean0 = np.array([gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
#         mean1 = np.array([-gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
#         means_init = np.array([mean0, mean1])
#         precisions_init = np.array(
#             [np.diag(1 / gmm_0.covariances_[0]), np.diag(1 / gmm_0.covariances_[0])])
#         # precisions_init = np.array([1 / gmm_0.covariances_[0], 1 / gmm_0.covariances_[0]])
#         precisions_init = np.diag(1 / gmm_0.covariances_[0])

#         gmm_RepCode = GaussianMixture(n_components=n_components, covariance_type='tied',
#                                       means_init=means_init, precisions_init=precisions_init, init_params="k-means++", random_state=42,
#                                       tol=tol, max_iter=max_iter, n_init=1)

#     gmm_RepCode.fit(process_complex_data(IQ_data))
#     return gmm_RepCode


# # def fit_RepCode_data(IQ_data, gmm_0, gmm_1, n_components):
# #     gmm_RepCode = GaussianMixture(n_components=n_components, covariance_type='full', 
# #                                   init_params="k-means++", tol=0.00001, max_iter=5000,
# #                                   n_init=2, random_state=42)
# #     gmm_RepCode.fit(process_complex_data(IQ_data))
# #     return gmm_RepCode


# def fit_RepCode_data_2(IQ_data, gmm_0, gmm_1):
#     gmm_2 = fit_RepCode_data(IQ_data, gmm_0, gmm_1, n_components=2)
#     gmm_3_above = fit_RepCode_data(IQ_data, gmm_0, gmm_1, n_components=3, where='above')
#     gmm_3_below = fit_RepCode_data(IQ_data, gmm_0, gmm_1, n_components=3, where='below')


#     aic2 = gmm_2.aic(process_complex_data(IQ_data))
#     aic3_above = gmm_3_above.aic(process_complex_data(IQ_data))
#     aic3_below = gmm_3_below.aic(process_complex_data(IQ_data))

#     aic_diff_above = abs(aic2 - aic3_above)
#     aic_diff_below = abs(aic2 - aic3_below)
#     aic_rel_diff_above = aic_diff_above / max(aic2, aic3_above)
#     aic_rel_diff_below = aic_diff_below / max(aic2, aic3_below)

#     if aic_rel_diff_above < 1e-4 and aic_rel_diff_below < 1e-4: # Heuristic threshold, should work 
#         print(f"AIC relative difference above: {aic_rel_diff_above}, below: {aic_rel_diff_below}")
#         return gmm_2
    
#     # return gmm_2
#     return gmm_2 if aic2 < aic3_above else gmm_3_above if aic3_above < aic3_below else gmm_3_below
#     # return gmm_3_above if aic3_above < aic3_below else gmm_3_below  



# # def fit_RepCode_data_2(IQ_data, gmm_0, gmm_1):
# #     gmm_2 = fit_RepCode_data(IQ_data, gmm_0, gmm_1, n_components=2)
# #     gmm_3 = fit_RepCode_data(IQ_data, gmm_0, gmm_1, n_components=3)

# #     aic2 = gmm_2.aic(process_complex_data(IQ_data))
# #     aic3 = gmm_3.aic(process_complex_data(IQ_data))

# #     aic_diff = abs(aic2 - aic3)
# #     aic_rel_diff = aic_diff / max(aic2, aic3)

# #     bic2 = gmm_2.bic(process_complex_data(IQ_data))
# #     bic3 = gmm_3.bic(process_complex_data(IQ_data))

# #     bic_diff = abs(bic2 - bic3)
# #     bic_rel_diff = bic_diff / max(bic2, bic3)

# #     # print(f"AIC 2 components: {aic2}, AIC 3 components: {aic3}")
# #     # print(f"BIC 2 components: {bic2}, BIC 3 components: {bic3}")
# #     # Print the relative differences
# #     print(f"AIC relative difference: {aic_rel_diff}, BIC relative difference: {bic_rel_diff}")

# #     if aic_rel_diff > 0.1 or bic_rel_diff > 0.1:
# #         if aic_rel_diff > 0.1:
# #             if bic_rel_diff > 0.1:
# #                 warnings.warn("AIC and BIC differences are both above 10%.")
# #             else:
# #                 warnings.warn("AIC difference is above 10%.")
# #         else:
# #             warnings.warn("BIC difference is above 10%.")

# #     return gmm_2 if aic2 < aic3 else gmm_3

# def fit_calib_data(IQ_data0, IQ_data1):
#     gmm_0 = fit_gmm_0_calib(IQ_data0)
#     gmm_1_1comp = fit_gmm_1_calib(IQ_data1, gmm_0, n_components=1)
#     gmm_1_2comp = fit_gmm_1_calib(IQ_data1, gmm_0, n_components=2)

#     aic1 = gmm_1_1comp.aic(process_complex_data(IQ_data1))
#     aic2 = gmm_1_2comp.aic(process_complex_data(IQ_data1))
#     # print(f"BIC 1 component: {bic1}, BIC 2 components: {bic2}")

#     return gmm_0, gmm_1_1comp if aic1 < aic2 else gmm_1_2comp


# def process_complex_data(data):
#     real_parts = data.real
#     imag_parts = data.imag
#     combined_data = np.column_stack([real_parts, imag_parts])
#     return combined_data


# def combine_gmms_equally(gmm0, gmm1):
#     assert gmm0.covariance_type == 'diag' and gmm0.n_components == 1, "gmm0 must have one component with diagonal covariance"

#     means0 = gmm0.means_
#     means1 = gmm1.means_
#     combined_means = np.vstack((means0, means1))
#     total_components = gmm0.n_components + gmm1.n_components
#     combined_weights = np.ones(total_components) / total_components

#     # Create a new GMM instance with combined parameters
#     new_gmm = GaussianMixture(
#         n_components=total_components, covariance_type='full')

#     # Manually set weights and means
#     new_gmm.weights_ = combined_weights
#     new_gmm.means_ = combined_means

#     # Handle covariances and compute Cholesky decompositions
#     if gmm1.covariance_type == 'diag':
#         combined_precisions = np.vstack(
#             [1 / gmm0.covariances_, 1 / gmm1.covariances_])
#         # Convert precisions to covariances for the full type
#         combined_covariances = [np.diag(1 / p) for p in combined_precisions]
#     elif gmm1.covariance_type == 'full':
#         cov0_full = np.diag(gmm0.covariances_[0])
#         combined_covariances = np.array([cov0_full] + list(gmm1.covariances_))

#     # Convert combined covariances back to precisions and calculate Cholesky decomposition
#     combined_precisions = np.array(
#         [np.linalg.inv(cov) for cov in combined_covariances])
#     new_gmm.precisions_cholesky_ = np.array(
#         [np.linalg.cholesky(p) for p in combined_precisions])

#     return new_gmm


# def plot_gmm_classifications_and_ellipses(IQ_data, gmm_0, gmm_1, dpi=100):
#     X = process_complex_data(IQ_data)

#     # Predictions from both models
#     predictions_0 = gmm_0.predict(X)
#     predictions_1 = gmm_1.predict(X)

#     # Apply any specific figure formatting
#     apply_formatting(dpi=dpi)
#     fig = plt.figure(figsize=(12, 4))

#     # Plotting classifications from gmm_0
#     plt.subplot(1, 2, 1)
#     scatter0 = plt.scatter(X[:, 0], X[:, 1], s=1,
#                            c=predictions_0, cmap='viridis', alpha=0.1)
#     plot_ellipses(gmm_0, plt.gca())
#     plt.title('Classification by gmm_0')
#     plt.xlabel('In-Phase [arb.]')
#     plt.ylabel('Quadrature [arb.]')

#     # Plotting classifications from gmm_1
#     plt.subplot(1, 2, 2)
#     scatter1 = plt.scatter(X[:, 0], X[:, 1], s=1,
#                            c=predictions_1, cmap='viridis', alpha=0.1)
#     plot_ellipses(gmm_1, plt.gca())
#     plt.title('Classification by gmm_1')
#     plt.xlabel('In-Phase [arb.]')
#     plt.ylabel('Quadrature [arb.]')

#     return fig



# def plot_ellipses(gmm, ax):
#     for mean in gmm.means_:
#         if gmm.covariance_type == 'full':
#             # Eigen-decomposition of the covariance matrix for each component
#             cov = gmm.covariances_
#         elif gmm.covariance_type == 'diag':
#             cov = np.diag(gmm.covariances_)
#             w = np.eye(len(cov))
#         else:  # This covers 'spherical' if necessary, handling it like diag for illustration
#             v = np.array([gmm.covariances_] * len(mean))
#             w = np.eye(len(v))

#         # Scale the eigenvalues to get the standard deviations
#         v = 2. * np.sqrt(2.) * np.sqrt(v)
#         u = w[:, 0]  # Eigenvector corresponding to the largest eigenvalue
#         angle = np.arctan2(u[1], u[0])  # Angle in radians
#         angle = np.degrees(angle)  # Convert to degrees for plotting

#         ell = Ellipse(xy=mean, width=v[0], height=v[1],
#                       angle=angle, color='red', fill=False)
#         ax.add_artist(ell)
#         ell.set_clip_box(ax.bbox)
#         ell.set_alpha(0.5)

# def plot_gmm_classifications_and_ellipses_comb(IQ_data, gmm_combined, dpi=100, title=''):
#     X = process_complex_data(IQ_data)
#     combined_predictions = gmm_combined.predict(X)

#     apply_formatting(dpi=dpi)
#     fig = plt.figure()
#     scatter = plt.scatter(X[:, 0], X[:, 1], s=1,
#                           c=combined_predictions, cmap='viridis', alpha=0.1)

#     plot_ellipses_comb(gmm_combined, plt.gca())

#     plt.xlabel('In-Phase [arb.]')
#     plt.ylabel('Quadrature [arb.]')
#     plt.title(title)

#     return fig


# def plot_ellipses_comb(gmm, ax):
#     if gmm.covariance_type == 'tied':
#         # In 'tied', all components share the same precision matrix, invert it just once
#         prec = gmm.precisions_cholesky_ @ gmm.precisions_cholesky_.T
#         cov = np.linalg.inv(prec)
#         v, w = np.linalg.eigh(cov)
#         v = 2. * np.sqrt(2.) * np.sqrt(v)  # Scale the eigenvalues for plotting
#         for mean in gmm.means_:
#             u = w[:, 0]  # Eigenvector corresponding to the largest eigenvalue
#             angle = np.arctan2(u[1], u[0])
#             angle = np.degrees(angle)  # Convert to degrees
#             ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color='red', fill=False)
#             ax.add_artist(ell)
#             ell.set_clip_box(ax.bbox)
#             ell.set_alpha(0.5)
#     else:
#         for mean, prec_chol in zip(gmm.means_, gmm.precisions_cholesky_):
#             if gmm.covariance_type == 'full':
#                 # Construct full precision matrix from its Cholesky decomposition
#                 prec = prec_chol @ prec_chol.T
#             elif gmm.covariance_type == 'diag':
#                 # For diagonal, the Cholesky decomposition is the square root of the diagonal elements
#                 prec = np.diag(prec_chol**2)
#             else:
#                 # This block handles the spherical case where the Cholesky factor is just the standard deviation
#                 prec = np.diag(np.full(gmm.means_.shape[1], prec_chol**2))

#             # Invert the precision matrix to get the covariance matrix
#             cov = np.linalg.inv(prec)
#             # Calculate eigenvalues and eigenvectors for the covariance matrix
#             v, w = np.linalg.eigh(cov)
#             v = 2. * np.sqrt(2.) * np.sqrt(v)  # Scale the eigenvalues for plotting
#             u = w[:, 0]  # Eigenvector corresponding to the largest eigenvalue
#             angle = np.arctan2(u[1], u[0])
#             angle = np.degrees(angle)  # Convert to degrees
#             ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color='red', fill=False)
#             ax.add_artist(ell)
#             ell.set_clip_box(ax.bbox)
#             ell.set_alpha(0.5) 