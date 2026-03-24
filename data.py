import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("D:/vs code/csvfiles/UpdatedResumeDataSet.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nCategories:\n", df['Category'].value_counts())
print("\nSample:\n", df.head(5))

plt.figure(figsize=(14, 5))
df['Category'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Resume Count per Job Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('category_distribution.png')
plt.show()
print("\n✅ complete! Chart saved.")