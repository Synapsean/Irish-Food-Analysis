import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

print("Fetching Irish data for Machine Learning...")
response = supabase.table("food_items") \
    .select("product_name, ingredients_text, nova_group") \
    .or_("countries_sold.ilike.%Ireland%,countries_sold.ilike.%en:ie%") \
    .neq("ingredients_text", "") \
    .limit(2000) \
    .execute()

df = pd.DataFrame(response.data)
print(f"Loaded {len(df)} products.")

# Found that a lot of products contained German ingredients (or German spelling of ingredients to be precise)
# Filtering out common German spelling
german_keywords = ['zucker', 'wasser', 'spiesesalz', 'weizenmehl']
df = df[~df['ingredients_text'].str.lower().apply(lambda x: any(k in x for k in german_keywords))]
initial_count = len(df)
print(f"Removed {initial_count - len(df)} non-English products. Remaining: {len(df)}")

# 2. VECTORIZATION (TF-IDF)
# Convert text to numbers. We ignore words that appear in >50% of docs (max_df)
# to filter out generic terms like "water" if they are too common.
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=1000)
X = vectorizer.fit_transform(df['ingredients_text'])

print(f"Matrix Shape: {X.shape} (Products x Unique Ingredients)")

# 3. CLUSTERING (K-MEANS)
# We ask for 5 clusters. You can change this number later.
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Assign the "Cluster ID" back to the original data
df['cluster'] = kmeans.labels_

# 4. EXPLAIN THE CLUSTERS
# What defines each group? Look at the "Centroids" (the center of the cluster).
print("\n--- CLUSTER INSIGHTS ---")
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(k):
    print(f"\nCluster {i}:")
    # Print the top 10 words that define this cluster
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"Top Keywords: {', '.join(top_terms)}")
    
    # Show 3 random examples from this cluster
    examples = df[df['cluster'] == i]['product_name'].sample(3).values
    print(f"Examples: {examples}")

# 5. VISUALISATION (PCA)
# Squash the 1000 dimensions down to 2 so we can plot it
pca = PCA(n_components=2)
coords = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=coords[:, 0], 
    y=coords[:, 1], 
    hue=df['cluster'], 
    palette='viridis',
    s=70,
    alpha=0.8
)
plt.title('Food Clusters: Visualising Ingredient Similarity')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster ID')
plt.tight_layout()
plt.savefig('cluster_analysis.png', dpi=300)
print("\nGraph saved to cluster_analysis.png")
#plt.show()

# 6. VALIDATION: DO CLUSTERS MATCH NOVA GROUPS?
print("\n--- NOVA GROUP BREAKDOWN BY CLUSTER ---")

# Clean the data: Filter out rows where NOVA is missing or 'unknown'
clean_df = df[df['nova_group'].notna() & (df['nova_group'] != 'unknown')]

# Create the Cross-Tabulation table
# Rows = The Clusters your AI found
# Columns = The Official NOVA Groups (1, 2, 3, 4)
crosstab = pd.crosstab(clean_df['cluster'], clean_df['nova_group'])
print(crosstab)

# VISUALIZE IT
plt.figure(figsize=(8, 5))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Reds')
plt.title('Validation: Do Clusters Predict Processing Level (NOVA)?')
plt.ylabel('Cluster ID (AI Generated)')
plt.xlabel('NOVA Group (Official Label)')

plt.tight_layout()
plt.savefig('nova_validation.png', dpi=300)
print("Validation graph saved to nova_validation.png")
plt.show()