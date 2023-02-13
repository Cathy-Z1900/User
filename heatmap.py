import numpy as np
import matplotlib
import matplotlib.pyplot as plt

gender = ["gender_1", "gender_2"]
age_range = ["age_range_0", "age_range_1", "age_range_2",
           "age_range_3", "age_range_4", "age_range_5", "age_range_6"]

data = np.array([[0,0.061,0.108,0.051,0.068,0.101,0.055
],
                    [0.333,0.121,0.087,0.132,0.115,0.09,0.038
],
                    ])
# =============================================
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    cbar=ax.figure.colorbar(im,ax=ax, **cbar_kw,orientation="horizontal", fraction=0.05)
    cbar.ax.set_xlabel(cbarlabel,size=12,rotation=0, va="top")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels,size=18,text='bold')

    ax.set_yticklabels(row_labels,size=18,text='bold')
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor",text='bold')
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

fig, ax = plt.subplots()

im, cbar = heatmap(data, gender, age_range, ax=ax,
                   cmap="Blues",cbarlabel='User explicit interest range')
texts = annotate_heatmap(im,size=18)
# ax.set_title("User explicit interest probability")
fig.tight_layout()

plt.show()
# ========================================================

gender = ["occupation_0", "ocuupation_1"]
age_range = ["location_0", "location_1", "location_2",
           "location_3", "location_4", "location_5", "location_6",'location_7',
             'location_8','location_9','location_10','location_11','location_12']

data = np.array([[0.142857143,0.026785714,0.033175355,0.02907489,0.040748899,0.02811245,
0.023255814,0.03649635,0.030508475,
0.021959459,0.023550725,0.035629454,0.032786885],
[0,0.066666667,0.016042781,0.023809524,
0,
0,
0,
0,
0.065789474,
0,
0,
0,
0

],
                    ])
# =============================================
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()


    im = ax.imshow(data, **kwargs)

    cbar=ax.figure.colorbar(im,ax=ax, **cbar_kw,orientation="horizontal", fraction=0.05)

    cbar.ax.set_xlabel(cbarlabel,size=12,rotation=0, va="top") ###modify cbar'title, x label and y

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels,size=14,text='bold')
    ax.set_yticklabels(row_labels,size=14,text='bold')

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor",text='bold')

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)


    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

fig, ax = plt.subplots()

im, cbar = heatmap(data, gender, age_range, ax=ax,
                   cmap="Blues",cbarlabel='User implicit interest range')
texts = annotate_heatmap(im,size=14)
# ax.set_title("User explicit interest probability")
fig.tight_layout()

plt.show()