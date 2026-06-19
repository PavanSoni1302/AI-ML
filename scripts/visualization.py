import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Age': [21, 22, 23, 24, 25],
    'Score': [80, 82, 88, 92, 95]
}

df = pd.DataFrame(data)

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Feature Correlation Map")
plt.show()