import pandas as pd
import plotly
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

df=pd.read_csv('all.csv')

X = df.drop('action_type', axis=1)
y = df['action_type']
y = y.map({'c':'click','t':'add_to_cart','b':'add_to_favorite','p':'purchase'})

cat_cols= X.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    print (f"col name : {col}, N Unique : {X[col].nunique()}")

for col in cat_cols:
    X[col]=X[col].astype('category')
    X[col]=X[col].cat.codes
print(X.head())
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_std = StandardScaler().fit_transform(X)
####TSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_std)
X_tsne_data = np.vstack((X_tsne.T, y)).T
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'action_type'])
print(df_tsne.head())
plt.figure(figsize=(16, 8))
sns.scatterplot(data=df_tsne, hue='action_type', x='Dim1', y='Dim2')
plt.show()
