"""
Irish Food Market Analysis Dashboard
Optimised for performance and smooth navigation.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from dotenv import load_dotenv
from supabase import create_client, Client

# Add src to path for import safety
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Fallback for Tokenizer
try:
    from tokenizer import parse_ingredient_list
except ImportError:
    def parse_ingredient_list(text):
        return [x.strip().lower() for x in str(text).split(',') if x.strip()]

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Irish Food Detective",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #169B62;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 20px;
    }
    .insight-box {
        background: linear-gradient(135deg, #1E3A2F 0%, #2D4A3E 100%);
        border-left: 5px solid #169B62;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: rgba(22, 155, 98, 0.1);
        padding: 10px;
        border-radius: 8px;
    }
    @keyframes skeleton-loading {
        0% { background-color: #1a1a2e; }
        50% { background-color: #16213e; }
        100% { background-color: #1a1a2e; }
    }
    .skeleton {
        animation: skeleton-loading 1.5s infinite;
        border-radius: 8px;
        height: 400px;
        margin: 10px 0;
    }
    .footer-text {
        font-size: 0.85rem;
        color: #888;
        text-align: center;
        padding: 10px 0;
    }
    .footer-text a {
        color: #169B62;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Load and clean data from Supabase."""
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url:
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        except:
            return pd.DataFrame()
    
    supabase: Client = create_client(url, key)
    
    response = supabase.table("food_items") \
        .select("product_name, ingredients_text, nova_group, sugar_100g, salt_100g, fat_100g, category_searched, brand") \
        .or_("countries_sold.ilike.%Ireland%,countries_sold.ilike.%en:ie%") \
        .neq("ingredients_text", "") \
        .limit(2000) \
        .execute()
    
    df = pd.DataFrame(response.data)
    
    if df.empty:
        return df
    
    # Cleaning Pipeline
    for col in ['sugar_100g', 'salt_100g', 'fat_100g']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Filter German products
    german_keywords = ['zucker', 'wasser', 'spiesesalz', 'weizenmehl']
    df = df[~df['ingredients_text'].str.lower().apply(
        lambda x: any(k in str(x) for k in german_keywords)
    )]
        
    # Flag soup concentrates (stock cubes, powders)
    concentrate_words = ['stock', 'pot', 'cube', 'bouillon', 'powder', 'instant', 'dry', 'mix']
    df['is_soup_concentrate'] = df.apply(
        lambda row: (row['category_searched'] == 'Soup') and 
                    any(w in str(row['product_name']).lower() for w in concentrate_words), 
        axis=1
    )
    
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_clustering(_df, n_clusters):
    """Run ML Pipeline (cached by cluster count)."""
    df = _df.copy()
    
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=1000)
    X = vectorizer.fit_transform(df['ingredients_text'])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Get top terms per cluster
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for i in range(n_clusters):
        top_indices = kmeans.cluster_centers_[i].argsort()[-10:][::-1]
        cluster_terms[i] = [terms[idx] for idx in top_indices]
    
    # PCA for visualisation
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X.toarray())
    df['pca_x'] = coords[:, 0]
    df['pca_y'] = coords[:, 1]
    
    return df, cluster_terms, kmeans.inertia_


@st.cache_data(show_spinner=False)
def get_ingredient_counts(_df):
    """Count all ingredients (cached)."""
    all_ingredients = []
    for text in _df['ingredients_text']:
        all_ingredients.extend(parse_ingredient_list(str(text)))
    return Counter(all_ingredients)

# --- MAIN APP ---
def main():
    # Header
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/45/Flag_of_Ireland.svg", width=50)
    st.sidebar.title("Food Detective")
    
    # Load Data First
    df = load_data()
    
    if df.empty:
        st.error("⚠️ Could not load data. Check your Supabase credentials in `.env`")
        st.code("SUPABASE_URL=your_url\nSUPABASE_KEY=your_key", language="bash")
        return
    
    st.sidebar.success(f"✅ {len(df)} products loaded")
    st.sidebar.divider()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate", 
        ["📊 Cluster Explorer", "🍬 Nutrition Insights", "🧪 Ingredients", "📈 Data", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    # Sidebar Footer
    st.sidebar.divider()
    st.sidebar.markdown("""
    <div class="footer-text">
        Built by <b>Sean</b><br>
        <a href="https://linkedin.com/in/YOUR-LINKEDIN" target="_blank">🔗 LinkedIn</a> • 
        <a href="https://github.com/Synapsean" target="_blank">💻 GitHub</a>
    </div>
    """, unsafe_allow_html=True)
    
    # --- PAGE 1: CLUSTER EXPLORER ---
    if page == "📊 Cluster Explorer":
        st.markdown('<p class="main-header">🧠 Market Segmentation</p>', unsafe_allow_html=True)
        st.caption("K-Means clustering groups products by ingredient similarity")
        
        n_clusters = st.sidebar.slider("Clusters (k)", 2, 10, 5, key="k_slider")
        
        with st.spinner("Running clustering..."):
            df_clustered, cluster_terms, inertia = run_clustering(df, n_clusters)
            
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.scatter(
                df_clustered, x='pca_x', y='pca_y', 
                color=df_clustered['cluster'].astype(str),
                hover_data=['product_name', 'category_searched'],
                title='Product Clusters (PCA Projection)',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=550, legend_title="Cluster")
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.metric("Inertia", f"{inertia:,.0f}", help="Lower = tighter clusters")
            st.divider()
            
            for cid, terms in cluster_terms.items():
                count = (df_clustered['cluster'] == cid).sum()
                with st.expander(f"**Cluster {cid}** ({count} items)"):
                    st.write("🏷️ " + ", ".join(terms[:5]))
                    samples = df_clustered[df_clustered['cluster'] == cid]['product_name'].head(3).tolist()
                    st.caption("Examples: " + " • ".join(samples))

    # --- PAGE 2: NUTRITION INSIGHTS ---
    elif page == "🍬 Nutrition Insights":
        st.markdown('<p class="main-header">🔬 Nutrition Analysis</p>', unsafe_allow_html=True)
        st.caption("Statistical analysis of salt and sugar content")
        
        # THE SALT WAR
        st.markdown("""
        <div class="insight-box">
        <b>🧂 The Hidden Salt Trap</b><br>
        Soup appears saltier than crisps—but this is skewed by concentrated stock cubes (15g+ salt/100g).
        Toggle below to see the real comparison.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### Controls")
            include_concentrates = st.toggle("Include Stock Cubes", value=False)
            
            soup_clean = df[(df['category_searched'] == 'Soup') & (~df['is_soup_concentrate'])]
            soup_all = df[df['category_searched'] == 'Soup']
            crisps = df[df['category_searched'] == 'Crisps']
            
            st.divider()
            st.metric("🍲 Real Soup (avg)", f"{soup_clean['salt_100g'].mean():.2f}g")
            st.metric("🥔 Crisps (avg)", f"{crisps['salt_100g'].mean():.2f}g")
            if include_concentrates:
                st.metric("📦 With Cubes (avg)", f"{soup_all['salt_100g'].mean():.2f}g", 
                         delta=f"+{soup_all['salt_100g'].mean() - soup_clean['salt_100g'].mean():.1f}g")

        with col2:
            plot_df = df[df['category_searched'].isin(['Soup', 'Crisps'])].copy()
            if not include_concentrates:
                plot_df = plot_df[~plot_df['is_soup_concentrate']]
                title = "Salt: Ready-to-Eat Soup vs Crisps"
            else:
                title = "Salt: All Soup Products (includes concentrates)"
                
            fig = px.box(
                plot_df, x="category_searched", y="salt_100g", 
                color="category_searched", points="outliers",
                title=title, hover_data=["product_name"],
                color_discrete_map={"Soup": "#4C78A8", "Crisps": "#F58518"}
            )
            fig.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        
        # Sugar by Category
        st.subheader("🍬 Sugar by Category")
        cat_sugar = df.groupby('category_searched')['sugar_100g'].mean().sort_values(ascending=True)
        fig2 = px.bar(x=cat_sugar.values, y=cat_sugar.index, orientation='h',
                      labels={'x': 'Avg Sugar (g/100g)', 'y': 'Category'},
                      color=cat_sugar.values, color_continuous_scale='Oranges')
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # --- PAGE 3: INGREDIENTS ---
    elif page == "🧪 Ingredients":
        st.markdown('<p class="main-header">🧪 Ingredient Analysis</p>', unsafe_allow_html=True)
        st.caption("Most common ingredients across Irish food products")
        
        counts = get_ingredient_counts(df)
        top_n = st.slider("Show top N ingredients", 10, 50, 20)
        
        top_df = pd.DataFrame(counts.most_common(top_n), columns=['Ingredient', 'Count'])
        
        fig = px.bar(
            top_df, x='Count', y='Ingredient', orientation='h',
            color='Count', color_continuous_scale='Greens'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=max(400, top_n * 25))
        st.plotly_chart(fig, use_container_width=True)
        
        # Search
        st.divider()
        search = st.text_input("🔍 Search for an ingredient:", placeholder="e.g., palm oil")
        if search:
            matches = df[df['ingredients_text'].str.lower().str.contains(search.lower(), na=False)]
            st.write(f"Found **{len(matches)}** products containing '{search}'")
            if not matches.empty:
                st.dataframe(
                    matches[['product_name', 'brand', 'category_searched']].head(20),
                    use_container_width=True, hide_index=True
                )

    # --- PAGE 4: RAW DATA ---
    elif page == "📈 Data":
        st.markdown('<p class="main-header">📊 Data Explorer</p>', unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            cats = st.multiselect("Filter by Category", df['category_searched'].unique().tolist())
        with col2:
            search = st.text_input("Search product name", "")
        
        filtered = df.copy()
        if cats:
            filtered = filtered[filtered['category_searched'].isin(cats)]
        if search:
            filtered = filtered[filtered['product_name'].str.lower().str.contains(search.lower(), na=False)]
        
        st.write(f"Showing **{len(filtered)}** of {len(df)} products")
        
        display_cols = ['product_name', 'brand', 'category_searched', 'sugar_100g', 'salt_100g', 'nova_group']
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
        
        # Download
        st.download_button(
            "📥 Download CSV", 
            filtered.to_csv(index=False), 
            "irish_food_data.csv", 
            "text/csv"
        )

    # --- PAGE 5: ABOUT ---
    elif page == "ℹ️ About":
        st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 The Mission
            
            This project answers a simple question: **What's really in Irish food?**
            
            Using data from 1,000+ products sold in Ireland, I built an end-to-end data science 
            pipeline that harvests, cleans, and analyses ingredient data — then applies 
            **unsupervised machine learning** to automatically detect patterns.
            
            ---
            
            ### 🧠 How It Works
            
            1. **Data Harvesting** — Products are fetched from the OpenFoodFacts API, filtered for Ireland
            2. **NLP Preprocessing** — Custom regex tokenizer handles nested ingredient lists
            3. **TF-IDF Vectorization** — Converts ingredient text into numerical features
            4. **K-Means Clustering** — Groups similar products without human labels
            5. **Validation** — Clusters are validated against official NOVA processing scores
            
            ---
            
            ### 🔬 Key Findings
            
            - **Salt & Sugar** appear in over 45% of all products
            - The model identified **5 distinct market segments** automatically
            - "Soup" category averages are skewed by concentrated stock cubes
            - AI-generated clusters strongly correlate with NOVA ultra-processing scores
            
            ---
            
            ### 🛠️ Tech Stack
            
            `Python` `Pandas` `scikit-learn` `Streamlit` `Plotly` `Supabase` `PostgreSQL`
            
            """)
        
        with col2:
            st.markdown("""
            ### 👤 About Me
            
            **Sean**  
            MSc Data Analytics Student
            
            I'm passionate about using data science 
            to uncover hidden patterns in everyday life.
            
            This project demonstrates:
            - ETL pipeline development
            - NLP text preprocessing  
            - Unsupervised ML (clustering)
            - Statistical inference
            - Interactive dashboards
            
            ---
            
            ### 📬 Get In Touch
            
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/seán-quinlan-phd)
            
            [![GitHub](https://img.shields.io/badge/GitHub-Synapsean-black?style=flat&logo=github)](https://github.com/Synapsean)
            
            ---
            
            ### 📂 Source Code
            
            This project is open source!  
            [View on GitHub →](https://github.com/Synapsean/Irish-Food-Analysis)
            """)
        
        # Fun stats
        st.divider()
        st.subheader("📊 Dataset Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Products", f"{len(df):,}")
        c2.metric("Categories", df['category_searched'].nunique())
        c3.metric("Unique Brands", df['brand'].nunique())
        c4.metric("Data Source", "OpenFoodFacts")


if __name__ == "__main__":
    main()