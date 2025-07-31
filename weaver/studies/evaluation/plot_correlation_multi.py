# Plot correlation between score and a variable for multiple processes

import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)


def plot_correlation_multi(events,
            categories,
            xsecweighting = False,
            outputdir = None,
            score_branch = None,
            score_bins = None,
            variables = None):

    # check arguments
    if score_branch is None: raise Exception('Must provide a score branch.')
    if variables is None: raise Exception('Must provide at least one variable.')
    if score_bins is None: score_bins = [0, 0.5, 1]
    
    # get scores
    scores = events[score_branch]

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

    # set line styles for different score bins
    linestyles = ['solid']
    for idx in range(len(score_bins)-2):
        linestyles.append( (0, (1, 1 + 1*idx)) )

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # loop over variables
    for varname, variable in variables.items():
        fig, ax = plt.subplots()

        # loop over categories
        for category_name, category_settings in categories.items():

            # temporary: skip additional binning for signal
            # (maybe later add as argument instead of hard-coding)
            this_score_bins = score_bins[:]
            if category_name=='HH': this_score_bins = [score_bins[0], score_bins[-1]]
        
            # loop over score bins
            for idx in range(len(this_score_bins)-1):
                minscore = this_score_bins[idx]
                maxscore = this_score_bins[idx+1]
                score_bin_label = '({:.2f} - {:.2f})'.format(minscore, maxscore)

                # get correlation variable in score bin
                mask = ((scores > minscore) & (scores < maxscore) & masks[category_name])
                this_values = events[variable['branch']][mask]
                this_weights = weights[mask]
            
                # make a histogram
                bins = variable['bins']
                hist = np.histogram(this_values, bins=bins, weights=this_weights)[0]
                norm = np.sum( np.multiply(hist, np.diff(bins) ) )
                if norm<1e-12: continue
                staterrors = np.sqrt(np.histogram(this_values, bins=bins, weights=np.square(this_weights))[0])
                ax.stairs(hist/norm, edges=bins,
                  color = category_settings['color'],
                  linestyle = linestyles[idx],
                  label = category_settings['label'] + ' ' + score_bin_label,
                  linewidth=2)
                ax.stairs((hist+staterrors)/norm, baseline=(hist-staterrors)/norm,
                        color = category_settings['color'],
                        edges=bins, fill=True, alpha=0.15)
        
        ax.set_xlabel(variable['label'], fontsize=12)
        ax.set_ylabel('Events (normalized)', fontsize=12)
        ax.set_ylim((0., ax.get_ylim()[1]*1.3))
        ax.set_title(f'Correlation between {varname} and classifier output score', fontsize=12)
        leg = ax.legend(fontsize=10)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
            lh._sizes = [30]
        fig.tight_layout()
        figname = os.path.join(outputdir, f'correlation_multi_{varname}_slices.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')
        plt.close()
