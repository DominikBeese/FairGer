from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

dfs = {d: pd.read_json(r'../../Data/Model Predicted Data/%s-Predictions.json' % d) for d in ['Frau', 'Migrant']}
dfs = {d: df[['id', 'year', 'month', 'day', 'category', 'keyword']] for d, df in dfs.items()}
dfs['Frau']['category'] = 'Woman'

cp = sns.color_palette()


# distribution of keywords (Pie Duo)
plt.figure(figsize=(5.4, 3.0))
plt.subplots_adjust(left=0.04, right=0.86, top=0.82, bottom=0.10)
for r, (d, df) in enumerate(dfs.items()):
	plt.subplot(1, 2, r+1)
	
	dft = df.copy(deep=True)
	dft = dft.groupby('keyword').size().sort_values(ascending=False).reset_index()
	dft['percentage'] = dft[0].div(dft.sum()[0])
	drop = dft[dft['percentage'] < 0.035]
	dft['keyword'] = ['$%s$' % s for s in dft['keyword']]
	
	dft = dft.drop(drop.index)
	dft = dft.append({'keyword': 'Other', 0: drop.sum()[0], 'percentage': drop.sum()['percentage']}, ignore_index=True)
	
	plt.pie(
		x=dft['percentage'],
		labels=dft['keyword'],
		colors=sns.color_palette('light:b_r', n_colors=len(dft)-1) + [(0.8,)*3],
		startangle=180,
		counterclock=False,
		pctdistance=0.8,
		autopct='%.0f%%',
		textprops={'size': 9}
	)
	
	plt.title({'Frau': 'Woman'}.get(d, d))

plt.suptitle('Distribution of Keywords', x=0.45)
plt.savefig('DistributionOfKeywords.pdf')
plt.show()
