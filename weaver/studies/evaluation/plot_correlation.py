import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)


def plot_correlation(events,
                categories,
                xsecweighting = False,
                outputdir = None,
                score_branch = None,
                variables = None,
                slices = None):

    # check arguments
    if score_branch is None: raise Exception('Must provide a score branch.')
    if variables is None: raise Exception('Must provide at least one variable.')
    
    # get scores
    scores = events[score_branch]
    if len(scores)==0: return

    # get weights
    weights = np.ones(len(scores))
    if xsecweighting:
        weights = np.multiply(events['lumiwgt'], np.multiply(events['genWeight'], events['xsecWeight']))

    # get mask for each category
    masks = {}
    for category_name, category_settings in categories.items():
        branch = category_settings['branch']
        mask = events[branch].astype(bool)
        masks[category_name] = mask

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # loop over categories
    for category_name, category_settings in categories.items():

        # loop over variables
        for varname, variable in variables.items():

            '''# make a scatter plot of the correlation
            cvar = events[varname]
            cscores = scores
            cweights = weights # note: weights not used for now in scatter plot
            if category_branch is not None:
                cvar = cvar[mask]
                cscores = cscores[mask]
                cweights = cweights[mask]
            fig, ax = plt.subplots()
            ax.scatter(cvar, cscores,
                color='dodgerblue', label=label, alpha=0.3, s=1)
            ax.set_xlabel(varname, fontsize=12)
            ax.set_ylabel('Classifier output score', fontsize=12)
            ax.set_title(f'Correlation between {varname} and classifier output score', fontsize=12)
            txt = ax.text(0.95, 0.3, label, fontsize=12,
                    ha='right', va='top', transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='white', alpha=0.5))
            if calculate_disco:
                txt = ax.text(0.95, 0.2, 'DisCo: {:.3f}'.format(dccoeffs[varname]), fontsize=12,
                        ha='right', va='top', transform=ax.transAxes)
                txt.set_bbox(dict(facecolor='white', alpha=0.5))
            figname = os.path.join(outputdir, f'correlation_{tag}_{varname}_scatter.png')
            fig.savefig(figname)
            print(f'Saved figure {figname}.')'''

            # same as above but in 2D histogram format instead of scatter plot
            this_values = events[variable['branch']][masks[category_name]]
            this_scores = scores[masks[category_name]]
            this_weights = weights[masks[category_name]]
            var_bins = variable['bins']
            score_bins = np.linspace(np.amin(this_scores), np.amax(this_scores), num=101)
            hist = np.histogram2d(this_scores, this_values, weights=this_weights,
                    bins=(score_bins, var_bins))[0]
            hist /= np.amax(hist)
            fig, ax = plt.subplots()
            im = ax.imshow(hist, cmap='plasma', interpolation='none', origin='lower', aspect='auto',
                        extent=(var_bins[0], var_bins[-1], score_bins[0], score_bins[-1]),
                        norm=mpl.colors.LogNorm(vmin=1e-2, vmax=1))
            fig.colorbar(im, ax=ax, label='Density')
            ax.set_xlabel(variable['label'], fontsize=12)
            ax.set_ylabel('Classifier output score', fontsize=12)
            ax.set_title(f'Correlation between {varname} and classifier output score', fontsize=12)
            txt = ax.text(0.95, 0.95, f'{category_settings["label"]} category', fontsize=12,
                      ha='right', va='top', transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='white', alpha=0.8))
            figname = os.path.join(outputdir,
                        f'correlation_{category_name}_{varname}_density.png')
            fig.savefig(figname)
            print(f'Saved figure {figname}.')
            plt.close()

            # plot correlation variables in slices of the score
            slices = [float(el) for el in slices]
        
            # loop over slices
            values_in_slices = []
            labels = []
            for idx in range(len(slices)-1):
                minscore = slices[idx]
                maxscore = slices[idx+1]

                # get correlation variable in slice 
                slicemask = ((scores > minscore) & (scores < maxscore))
                totalmask = (slicemask & masks[category_name])
                this_values = events[variable['branch']][totalmask]
                this_weights = weights[totalmask]
                values_in_slices.append((this_values, this_weights))
                labels.append('{:.2f} < score < {:.2f}'.format(minscore, maxscore))

            # make a plot
            fig, ax = plt.subplots()
            bins = variable['bins']
            cmap = plt.get_cmap('cool', len(slices))
            for idx, (cslice, clabel) in enumerate(zip(values_in_slices, labels)):
                hist = np.histogram(cslice[0], bins=bins, weights=cslice[1])[0]
                norm = np.sum( np.multiply(hist, np.diff(bins) ) )
                staterrors = np.sqrt(np.histogram(cslice[0], bins=bins, weights=np.square(cslice[1]))[0])
                ax.stairs(hist/norm, edges=bins,
                  color=cmap(idx), label=clabel, linewidth=2)
                ax.stairs((hist+staterrors)/norm, baseline=(hist-staterrors)/norm,
                        color=cmap(idx), edges=bins, fill=True, alpha=0.15)
            ax.set_xlabel(variable['label'], fontsize=12)
            ax.set_ylabel('Events (normalized)', fontsize=12)
            ax.set_title(f'Correlation between {varname} and classifier output score', fontsize=12)
            txt = ax.text(0.05, 0.95, f'{category_settings["label"]} category', fontsize=12,
                    ha='left', va='top', transform=ax.transAxes)
            txt.set_bbox(dict(facecolor='white', alpha=0.5))
            leg = ax.legend(fontsize=12)
            for lh in leg.legend_handles:
                lh.set_alpha(1)
                lh._sizes = [30]
            fig.tight_layout()
            figname = os.path.join(outputdir,
                        f'correlation_{category_name}_{varname}_slices.png')
            fig.savefig(figname)
            print(f'Saved figure {figname}.')
            plt.close()

    # close all figures
    plt.close()
