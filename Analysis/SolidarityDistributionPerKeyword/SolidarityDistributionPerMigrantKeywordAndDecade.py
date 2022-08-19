from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

df = pd.read_json(r'../../Data/Model Predicted Data/Migrant-Predictions.json')
df = df[['id', 'year', 'keyword', 'label']]

cp = sns.color_palette()

# distribution of solidarity per keyword and decade (Line Grid)
order = df.groupby('keyword').size().sort_values(ascending=False).index
dft = df.copy(deep=True)
dft['decade'] = dft['year']//10*10+5
dft = dft.groupby(['keyword', 'label', 'decade']).size().reset_index().pivot(columns=['keyword', 'label'], index='decade', values=0).fillna(0)
for keyword in order:
	if keyword not in dft: continue
	dft[keyword] = dft[keyword][dft[keyword].sum(axis=1)>=5] # decades with enough data only
	dft[keyword] = dft[keyword].div(dft[keyword].sum(axis=1), axis=0)
	for c in dft[keyword].columns: dft[(keyword, 'smooth-%d'%c)] = gaussian_filter1d(dft[keyword][c].interpolate(limit_area='inside'), sigma=1, truncate=1)

fig, axes = plt.subplots(figsize=(10.0, 4.0), ncols=8, nrows=4)
axes = [x for y in [*axes] for x in y]
for i, keyword in enumerate(order):
	ax, show_xlabel, show_ylabel = axes[i], i>=len(order)-8, i%8==0
	
	if keyword in dft and 0 in dft[keyword]:
		ax.bar(dft[keyword][0].index, dft[keyword][0].values, width=7, color=cp[2], alpha=0.4, linewidth=0)
		ax.plot(dft[keyword]['smooth-0'], color=cp[2])
	if keyword in dft and 1 in dft[keyword]:
		ax.bar(dft[keyword][1].index, dft[keyword][1].values, bottom=dft[keyword][0].fillna(0).values, width=7, color=cp[3], alpha=0.4, linewidth=0)
		ax.plot(dft[keyword]['smooth-0']+dft[keyword]['smooth-1'], color=cp[3])
	
	ax.set_xlim((1867, 2022))
	ax.set_ylim((0, 1.0))
	
	ax.tick_params(labelsize=10)
	ax.set_xticks([1900, 1950, 2000])
	ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
	
	ax.set_title('$%s$' % {'Sowjetzonenflüchtlinge': 'Sowjetzonen…', 'Bürgerkriegsflüchtlinge': 'Bürgerkriegs…'}.get(keyword, keyword), fontsize=9, pad=2)
	if show_xlabel: ax.set_xlabel('Year')
	else: ax.set_xlabel(None); ax.tick_params(labelbottom=False)
	if show_ylabel: ax.set_ylabel('Percentage', fontsize=11.5)
	else: ax.set_ylabel(None); ax.tick_params(labelleft=False)
for i in range(len(order), 32): fig.delaxes(axes[i])

plt.subplots_adjust(left=0.07, right=0.99, top=0.885, bottom=0.12, wspace=0.08, hspace=0.30)
plt.suptitle('Solidarity Distribution per Keyword and Decade', fontsize='medium')
plt.savefig('SolidarityDistributionPerMigrantKeywordAndDecade.pdf')
plt.show()
