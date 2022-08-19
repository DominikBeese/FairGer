from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

dfs = {d: pd.read_json(r'../../Data/Model Predicted Data/%s-Predictions.json' % d) for d in ['Frau', 'Migrant']}
dfs = {d: df[['id', 'year', 'label']] for d, df in dfs.items()}

cp = sns.color_palette()

# solidarity distribution (Stacked Line Duo)
plt.figure(figsize=(7.2, 3.0)); plt.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.83)
for r, (d, df) in enumerate(dfs.items()):
	ax = plt.subplot(1, 2, r+1)
	
	dft = df.copy(deep=True)
	dft['decade'] = dft['year']//10*10+5
	dft = dft.groupby(['label', 'decade']).size().reset_index().pivot(columns='label', index='decade', values=0).fillna(0)
	dft = dft[dft.sum(axis=1)>=5]
	dft = dft.div(dft.sum(axis=1), axis=0)
	for c in dft.columns: dft['smooth-%s'%c] = gaussian_filter1d(dft[c].interpolate(limit_area='inside'), sigma=1, truncate=1)
	
	plt.bar(dft[0].index, dft[0].values, width=7, color=cp[2], alpha=0.4, linewidth=0)
	plt.plot(dft['smooth-0'], color=cp[2], label='Solidarity')
	if d in ['Migrant']:
		plt.bar(dft[1].index, dft[1].values, bottom=dft[0].values, width=7, color=cp[3], alpha=0.4, linewidth=0)
		plt.plot(dft['smooth-0']+dft['smooth-1'], color=cp[3], label='Anti-solidarity')
	
	ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
	plt.title({'Frau': 'Woman'}.get(d, d))
	plt.xlabel('Year')
	if r == 0: plt.ylabel('Percentage of Solidarity')
	plt.ylim((0, None))
	plt.legend(loc='upper left')

plt.suptitle('Solidarity Distribution per Decade', x=0.542)
plt.savefig('SolidarityDistributionPerDecade.pdf')
plt.show()
