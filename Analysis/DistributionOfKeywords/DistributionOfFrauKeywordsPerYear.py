from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

df = pd.read_json(r'../../Data/Model Predicted Data/Frau-Predictions.json')
df = df[['id', 'year', 'month', 'day', 'category', 'keyword']]
df['category'] = 'Woman'

cp = sns.color_palette()


# distribution of keywords per year (Line Grid)
dft = df.copy(deep=True)
dft = dft.groupby(['keyword', 'year']).size().reset_index().pivot(columns='keyword', index='year', values=0).fillna(0)
dft = dft.reindex(range(min(dft.index), max(dft.index)+1)).fillna(0)
order = dft.sum(axis=0).sort_values(ascending=False).index
dft = dft.div(dft.sum(axis=0), axis=1)
for c in dft.columns: dft['smooth-%s'%c] = gaussian_filter1d(dft[c], sigma=3)

fig, axes = plt.subplots(figsize=(10.0, 4.0), ncols=6, nrows=3)
axes = [x for y in [*axes] for x in y]
for i, keyword in enumerate(order):
	ax, show_xlabel, show_ylabel = axes[i], i>=6*(3-1), i%6==0
	
	ax.plot(dft['smooth-%s' % keyword], color=cp[0])
	ax.bar(dft[keyword].reset_index()['year'], dft[keyword].reset_index()[keyword], width=0.7, color=cp[7], alpha=0.4, linewidth=0)
	
	ax.set_xlim((1867, 2022))
	ax.set_ylim((0, 0.13))
	
	ax.tick_params(labelsize=10)
	ax.set_xticks([1900, 1950, 2000])
	ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
	
	ax.set_title('$%s$' % keyword, fontsize=10, pad=2)
	if show_xlabel: ax.set_xlabel('Year')
	else: ax.set_xlabel(None); ax.tick_params(labelbottom=False)
	if show_ylabel: ax.set_ylabel('Popularity')
	else: ax.set_ylabel(None); ax.tick_params(labelleft=False)
for i in range(len(order), 18): fig.delaxes(axes[i])

plt.subplots_adjust(left=0.07, right=0.99, top=0.885, bottom=0.12, wspace=0.08, hspace=0.30)
plt.suptitle('Distribution of Keywords per Year', fontsize='medium')
plt.savefig('DistributionOfFrauKeywordsPerYear.pdf')
plt.show()
