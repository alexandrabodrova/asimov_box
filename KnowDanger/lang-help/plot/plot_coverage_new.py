import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


sns.set(font_scale=6)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})
#
# names = ['KnowNo', 'Simple Set', 'Ensemble Set', 'Prompt Set', 'Binary']
names = ['Attribute', 'Numeric', 'Spatial']
target = 0.85
# att_ambiguity = np.abs(np.array([0.86, 0.89, 0.86, 0.88, 0.76]) - target)
# num_ambiguity = np.abs(np.array([0.88, 0.82, 0.70, 0.74, 0.69]) - target)
# # syn_ambiguity = [0.86, 1.00, 0.90, 0.90, 0.90]
# spa_ambiguity = np.abs(np.array([0.88, 0.93, 0.82, 0.29, 1.00]) - target)

knowno = np.abs(np.array([0.86, 0.88, 0.88]) - target)
simple_set = np.abs(np.array([0.89, 0.82, 0.93]) - target)
ensemble_set = np.abs(np.array([0.86, 0.70, 0.82]) - target)
prompt_set = np.abs(np.array([0.88, 0.74, 0.29]) - target)
binary = np.abs(np.array([0.76, 0.69, 1.00]) - target)
no_help = np.abs(np.array([0.79, 0.565, 0.60]) - target)

# each ambiguity is the value for the name in each ambiguity category
# now make a bar plots showing the values for each category separately, so in three groups from left to right
# first group is the values for each name in att_ambiguity
# second group is the values for each name in num_ambiguity
# third group is the values for each name in syn_ambiguity

# make a dataframe with the values
df = pd.DataFrame(
    {
        # 'Attribute': att_ambiguity,
        # 'Numeric': num_ambiguity,
        # # 'Syntactic': syn_ambiguity,
        # 'Spatial': spa_ambiguity,
        'KnowNo': knowno,
        'Simple Set': simple_set,
        'Ensemble Set': ensemble_set,
        'Prompt Set': prompt_set,
        'Binary': binary,
        'No Help': no_help,
    },
    index=names
)

# plot with three greyscales
ax = df.plot.bar(
    rot=0,
    color=[
        np.array([241, 141, 0]) / 255,
        np.array([31, 119, 180]) / 255,
        np.array([128, 128, 128]) / 255,
        # np.array([188, 189, 33]) / 255,
        np.array([137, 138, 12]) / 255,
        # np.array([21, 190, 207]) / 255,
        np.array([4, 130, 143]) / 255,
        np.array([79, 72, 117]) / 255,
    ],
    width=0.8,
)

# set labels
# ax.set_xlabel("Name")
# ax.set_ylabel("Plan Success")
# ax.set_ylabel("Target coverage error")
# set legend
ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.05, 1.0),
    ncol=1,
    fancybox=True,
    shadow=False,
    fontsize=55,
    frameon=False,
)

# remove background color
ax.set_facecolor('white')

# set title
# ax.set_title("Ambiguity per name")
# set x ticks
# ax.set_xticklabels(names)
# ax.set_ylim(0.8, 1.0)
# ax.tick_params(labelbm=False)

# add a dashed line at y=0.85
# plt.axhline(y=0.85, color='black', linestyle='--', linewidth=1)

# change figure size
plt.gcf().set_size_inches(18, 20)

# # set y ticks
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])

ax.tick_params(which='minor', length=10, width=10, color='black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(which="both", bottom=True, left=True)

# legend off
# ax.get_legend().remove()

# save figure
plt.savefig("coverage_error_v2_legend.png", dpi=600, bbox_inches='tight')
