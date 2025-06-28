import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Netflix Life Impact Dataset (NLID) (1).csv")

# Clean and transform
df.drop_duplicates(inplace=True)
df['Genre'] = df['Genre'].str.strip().str.title()
df['Recommendation %'] = df['Suggested to Friends/Family (Y/N %)'].str.extract(r'(\d+)%').astype(int)

# Convert time to total minutes
time_parts = df['Minute of Life-Changing Insight'].str.extract(r'(\d+):(\d+)')
df['Insight_Minute'] = time_parts.apply(
    lambda row: int(row[0]) * 60 + int(row[1]) if pd.notnull(row[0]) and pd.notnull(row[1]) else None,
    axis=1
)

# -------------------- PLOT 1: Genre Count --------------------
genre_counts = df['Genre'].value_counts()
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='barh', color='skyblue')
plt.title("Number of Movies per Genre")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# -------------------- PLOT 2: Average Rating by Genre --------------------
avg_rating_by_genre = df.groupby('Genre')['Average Rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
avg_rating_by_genre.plot(kind='bar', color='lightgreen')
plt.title("Average Rating by Genre")
plt.ylabel("Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------- PLOT 3: Top 5 Most Reviewed Movies --------------------
top_reviewed = df.sort_values(by='Number of Reviews', ascending=False).head(5)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_reviewed, x="Movie Title", y="Number of Reviews", palette="Blues_d")
plt.title("Top 5 Most Reviewed Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------- PLOT 4: Rating vs Recommendation % --------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Average Rating", y="Recommendation %", hue="Genre", s=100)
plt.title("Rating vs Recommendation %")
plt.tight_layout()
plt.show()
