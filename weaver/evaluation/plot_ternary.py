# Plot scores in a ternary plot

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpltern


def plot_ternary(events,
            categories,
            outputdir = None):

    # check number of categories
    if len(categories)!=3:
        raise Exception('This function is only well defined for three categories.')
    category_names = list(categories.keys())

    # get mask for each category
    cat_masks = {}
    for category_name, category_settings in categories.items():
        branch = category_settings['label_branch']
        mask = events[branch].astype(bool)
        cat_masks[category_name] = mask

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='ternary')

    # loop over categories
    for category_name, category_settings in list(categories.items())[::-1]:
        cat_mask = cat_masks[category_name]

        # get the three scores
        scores = {}
        for category_name in category_names:
            score_branch = categories[category_name]['score_branch']
            scores[category_name] = events[score_branch][cat_mask]
            
        # make a scatter plot
        scores_1 = scores[category_names[0]]
        scores_2 = scores[category_names[1]]
        scores_3 = scores[category_names[2]]
        ax.scatter(scores_1, scores_2, scores_3,
                  color = category_settings['color'],
                  label = category_settings['label'],
                  alpha = 0.05,
                  s = 2)

    ax.grid()
    ax.set_tlabel(category_names[0] + ' score', fontsize=12)
    ax.set_llabel(category_names[1] + ' score', fontsize=10)
    ax.set_rlabel(category_names[2] + ' score', fontsize=10)
    leg = ax.legend(fontsize=10)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
        lh._sizes = [30]
    fig.tight_layout()
    figname = os.path.join(outputdir, f'scatter.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')
