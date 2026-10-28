"""Display the existing conditional fitting slices as a maximization problem.

This module changes only the sign and presentation of the frozen objective.
The displayed quantity is a scaled pseudo log-likelihood, not a likelihood
calculated from measured discrete choices. See the manuscript captions.
"""
import json
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


def install(f, renderer):
    """Install after the legacy and reference appearance layers."""
    renderer['HEIGHT_MM'].update({'FigureS4':145, 'FigureS5':180, 'FigureS6':195})

    def landscapes(bins):
        fits = f.read('corrected_expected_model_fits')
        best = fits[(fits.model == 'simple') & fits.converged].sort_values(['cross_entropy', 'MouseID'])
        mid = best.iloc[len(best) // 2].MouseID
        group = bins[bins.MouseID == mid].sort_values('Time_start')
        zt = group.ZT.to_numpy(float)
        elapsed = group.elapsed_h.to_numpy(float)
        index = group.bin_index.to_numpy(np.int64)
        events = f.allocate(group.total_g, 1000)
        weights = group.total_g.to_numpy() / group.total_g.sum()
        observed = group.preference.to_numpy()
        eligible = np.isfinite(observed)
        labels = [r'$\log_{10}\alpha$', r'$\log\beta_0$', r'$r_0$',
                  r'$A_r$', r'$\phi$ (rad)', r'$A_\beta$']
        manifest, checks = [], []

        for name, model in [('FigureS4', 'simple'), ('FigureS5', 'reward'), ('FigureS6', 'full')]:
            row = fits[(fits.MouseID == mid) & fits.model.eq(model)].iloc[0]
            parameters = np.array(json.loads(row.parameters))
            bounds = np.array(f.bounds_for(model))
            count = len(parameters)

            def ce(candidate, family=model):
                probability = np.clip(f.predict(family, candidate, zt, elapsed, index, events), 1e-10, 1-1e-10)
                return -np.sum(weights[eligible] * (observed[eligible] * np.log(probability[eligible])
                               + (1-observed[eligible]) * np.log1p(-probability[eligible])))

            reference = ce(parameters)
            diagonal, surfaces = {}, {}
            inversion_error = 0.
            for i in range(count):
                x = np.linspace(*bounds[i], 60)
                values = []
                for value in x:
                    candidate = parameters.copy(); candidate[i] = value
                    old_delta_loss = 1000 * (ce(candidate) - reference)
                    delta_ell = 1000 * (reference - ce(candidate))
                    inversion_error = max(inversion_error, abs(delta_ell + old_delta_loss))
                    values.append(delta_ell)
                diagonal[i] = (x, np.asarray(values))
                for j in range(i):
                    x = np.linspace(*bounds[j], 25); y = np.linspace(*bounds[i], 25)
                    z = np.zeros((25, 25))
                    for a, vertical in enumerate(y):
                        for b, horizontal in enumerate(x):
                            candidate = parameters.copy(); candidate[i] = vertical; candidate[j] = horizontal
                            old_delta_loss = 1000 * (ce(candidate) - reference)
                            z[a, b] = 1000 * (reference - ce(candidate))
                            inversion_error = max(inversion_error, abs(z[a, b] + old_delta_loss))
                    surfaces[i, j] = z

            all_values = np.concatenate([z.ravel() for z in surfaces.values()])
            # Positive values really improve upon the reference and must remain
            # visible. Only the *color* clips values below -100; data are intact.
            upper = max(0., float(all_values.max()))
            norm = Normalize(vmin=-100., vmax=upper)
            fig, axes = f.plt.subplots(count, count, figsize=(175/25.4, renderer['HEIGHT_MM'][name]/25.4))
            fig.subplots_adjust(left=.105, right=.92, top=.945, bottom=.105,
                                hspace=.55, wspace=.64)
            image = None
            for i in range(count):
                for j in range(count):
                    axis = axes[i, j]
                    if j > i:
                        axis.axis('off'); continue
                    if i == j:
                        x, y = diagonal[i]
                        axis.plot(x, y, color=f.MODEL_COLOR[model], lw=1.6)
                        axis.axhline(0, color='.55', lw=.8, ls=':')
                        axis.axvline(parameters[i], color='#aa3030', ls='--', lw=1.0)
                        axis.set_xlim(*bounds[i])
                        if i:
                            axis.yaxis.tick_right()
                        else:
                            axis.set_ylabel('Δ log-likelihood')
                        axis.yaxis.set_major_locator(MaxNLocator(2))
                    else:
                        image = axis.imshow(surfaces[i, j], origin='lower',
                                            extent=[*bounds[j], *bounds[i]], aspect='auto',
                                            cmap='viridis', norm=norm, interpolation='nearest')
                        axis.plot(parameters[j], parameters[i], 'r*', ms=5,
                                  clip_on=False, zorder=5)
                        if j == 0:
                            axis.set_ylabel(labels[i])
                        else:
                            axis.tick_params(labelleft=False)
                    axis.xaxis.set_major_locator(MaxNLocator(2))
                    if i == count-1:
                        axis.set_xlabel(labels[j])
                    else:
                        axis.tick_params(labelbottom=False)

            # Use otherwise empty upper-triangle space for a compact key.
            color_bounds = [.57, .80, .30, .018] if count == 3 else [.65, .84, .25, .015]
            color_axis = fig.add_axes(color_bounds)
            colorbar = fig.colorbar(image, cax=color_axis, orientation='horizontal', extend='min')
            colorbar.set_label('Δ log-likelihood', labelpad=2)
            colorbar.set_ticks([-100, -50, 0] + ([upper] if upper > 5 else []))

            if model == 'simple':
                axis = axes[0, 2]; axis.axis('on')
                bias = np.array(json.loads(fits[(fits.MouseID == mid) & fits.model.eq('bias')].iloc[0].parameters))
                x = np.linspace(-4, 4, 160)
                reference_bias = ce(bias, 'bias')
                y = np.asarray([1000*(reference_bias-ce(np.array([value]), 'bias')) for value in x])
                axis.plot(x, y, color=f.GRAY, lw=1.6)
                axis.axhline(0, color='.55', lw=.8, ls=':')
                axis.axvline(bias[0], color='#aa3030', ls='--', lw=1.)
                axis.set(xlabel='Bias intercept', ylabel='Δ log-likelihood', title='Bias')
                axis.xaxis.set_major_locator(MaxNLocator(2)); axis.yaxis.set_major_locator(MaxNLocator(2))
                # The Bias curve occupies the upper-right cell; put the key in
                # the unused middle-right cell instead.
                color_axis.set_position([.70, .49, .23, .018])

            checks.append(dict(figure=name, model=model, MouseID=mid,
                               reference_cross_entropy=float(reference),
                               reference_delta_log_likelihood=0.,
                               maximum_absolute_sign_inversion_error=float(inversion_error),
                               surface_minimum_delta_log_likelihood=float(all_values.min()),
                               surface_maximum_delta_log_likelihood=float(all_values.max()),
                               color_lower_limit=-100., color_upper_limit=upper,
                               positive_values_preserved=True,
                               diagonal_points_per_parameter=60, surface_grid_shape=[25,25]))
            assert inversion_error == 0
            f.save(fig, name)
            manifest.append(dict(figure=name, MouseID=mid, model=model,
                                 selection='Median cross-entropy among converged Simple fits, same session for all landscapes'))
        # This remains the original selection table, never a new fitted result.
        f.save_table(f.pd.DataFrame(manifest), 'landscape_selection')
        (renderer['DEST']/'audit'/'likelihood_surface_sign_check.json').write_text(
            json.dumps({'definition':'ell = -1000 * intake-weighted cross entropy; delta ell = ell(candidate) - ell(primary fit)',
                        'interpretation':'Scaled pseudo log-likelihood; larger is better. Unobserved event count is assumed.',
                        'checks':checks}, indent=2))

    f.landscapes = landscapes
