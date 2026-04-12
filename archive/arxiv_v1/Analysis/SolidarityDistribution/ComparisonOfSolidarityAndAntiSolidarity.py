from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sns.set_theme(style='whitegrid')

df = pd.read_json(r'../../Data/Model Predicted Data/Migrant-Predictions.json')
df = df[['id', 'year', 'label']]

cp = sns.color_palette()

# comparison of solidarity and anti-solidarity (Lineplot)
fig = plt.figure(figsize=(7.2*0.55, 3.0))
plt.subplots_adjust(left=0.17, right=0.99, bottom=0.16, top=0.83)

dft = df.copy(deep=True)
dft['decade'] = dft['year']//10*10+5
dft = dft.groupby(['label', 'decade']).size().reset_index().pivot(columns='label', index='decade', values=0).fillna(0)
dft = dft[dft.sum(axis=1)>=5]
dft = dft.div(dft.sum(axis=1), axis=0)
dft['0/1'] = dft[0] / dft[1]
dft['0-1'] = dft[0] - dft[1]
for c in dft.columns: dft['smooth-%s'%c] = gaussian_filter1d(dft[c].interpolate(limit_area='inside'), sigma=1, truncate=1)

plt.bar(dft['0-1'].index, dft['0-1'].values, width=7, alpha=0.4, linewidth=0, color=cp[7])
plt.plot(dft['smooth-0-1'], color='black')

fig.axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
plt.title('Migrant')
plt.xlabel('Year')
plt.ylabel('Difference of Solidarity', x=2)
fig.axes[0].yaxis.set_label_coords(-0.15, 0.52)

plt.suptitle('Comparison of Solidarity and Anti-solidarity', x=0.5)
plt.savefig('ComparisonOfSolidarityAndAntiSolidarity.pdf')
plt.show()
