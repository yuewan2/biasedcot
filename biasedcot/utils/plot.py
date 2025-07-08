import seaborn as sns
import matplotlib.pyplot as plt


def jointplot(tr, x, y, hue=None, reg=False, save_dir='', label_params={}):
    if hue is not None:
        g = sns.JointGrid(data=tr, x=x, y=y)
        sns.scatterplot(data=tr, x=x, y=y, hue=hue, s=50, ax=g.ax_joint, alpha=0.5)

        sns.histplot(data=tr, x=x, ax=g.ax_marg_x, bins=10, kde=True)
        sns.histplot(data=tr, y=y, ax=g.ax_marg_y, bins=10, kde=True)

        if reg:
            sns.regplot(data=tr, x=x, y=y, scatter=False, ax=g.ax_joint)

        if 'xlabel' in label_params:
            x = label_params['xlabel']
        if 'ylabel' in label_params:
            y = label_params['ylabel']
        if 'legend_title' in label_params:
            hue = label_params['legend_title']

        g.set_axis_labels(x, y, fontsize=14)
        if 'legend_labels' in label_params:
            g.ax_joint.legend(title=hue, loc='upper left', labels=label_params['legend_labels'], fontsize=10)
        else:
            g.ax_joint.legend(title=hue, loc='upper left', fontsize=10)
    else:
        g = sns.jointplot(x=tr[x], y=tr[y], kind="reg", marginal_kws=dict(bins=10))
        if 'xlabel' in label_params:
            x = label_params['xlabel']
        if 'ylabel' in label_params:
            y = label_params['ylabel']
        g.set_axis_labels(x, y, fontsize=14)

    if save_dir:
        plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    plt.show()