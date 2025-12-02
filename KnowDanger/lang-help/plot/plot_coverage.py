import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


sns.set(font_scale=6)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})
#
names = ['KnowNo', 'Simple Set', 'Ensemble Set', 'Prompt Set', 'Binary']
target = 0.85
att_ambiguity = np.abs(np.array([0.86, 0.89, 0.86, 0.88, 0.76]) - target)
num_ambiguity = np.abs(np.array([0.88, 0.82, 0.70, 0.74, 0.69]) - target)
# syn_ambiguity = [0.86, 1.00, 0.90, 0.90, 0.90]
spa_ambiguity = np.abs(np.array([0.88, 0.93, 0.82, 0.29, 1.00]) - target)

# each ambiguity is the value for the name in each ambiguity category
# now make a bar plots showing the values for each category separately, so in three groups from left to right
# first group is the values for each name in att_ambiguity
# second group is the values for each name in num_ambiguity
# third group is the values for each name in syn_ambiguity

# make a dataframe with the values
df = pd.DataFrame(
    {
        'Attribute': att_ambiguity,
        'Numeric': num_ambiguity,
        # 'Syntactic': syn_ambiguity,
        'Spatial': spa_ambiguity,
    },
    index=names
)

# plot with three greyscales
ax = df.plot.bar(rot=0, color=['#647C90', '#E2DED0', '#746C70'])

# set labels
# ax.set_xlabel("Name")
# ax.set_ylabel("Plan Success")
ax.set_ylabel("Target coverage error")
# set legend
ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.05, 1.0),
    ncol=1,
    fancybox=True,
    shadow=False,
    fontsize=60,
    frameon=False,
)

# remove background color
ax.set_facecolor('white')

# set title
# ax.set_title("Ambiguity per name")
# set x ticks
# ax.set_xticklabels(names)
# ax.set_ylim(0.8, 1.0)
ax.tick_params(labelbottom=False)

# add a dashed line at y=0.85
# plt.axhline(y=0.85, color='black', linestyle='--', linewidth=1)

# change figure size
plt.gcf().set_size_inches(18, 20)

# # set y ticks
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])

# save figure
plt.savefig("coverage_error.png", dpi=600, bbox_inches='tight')
