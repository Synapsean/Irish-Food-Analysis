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

# Import recommender module
try:
    from recommender import (
        find_healthier_alternatives,
        get_product_analysis,
        get_processing_badge,
        calculate_processing_score,
        count_e_numbers,
        find_concerning_additives
    )
    RECOMMENDER_AVAILABLE = True
except ImportError:
    RECOMMENDER_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Irish Food Detective",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>    .visitor-counter {
        font-size: 0.8rem;
        color: #888;
        text-align: right;
        padding: 5px 15px;
    }    .main-header {
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


@st.cache_data(show_spinner=False)
def get_dashboard_metrics(_df):
    """Calculate key dashboard metrics (cached)."""
    metrics = {}
    
    # Calculate processing scores for all products
    df_with_scores = _df.copy()
    df_with_scores['processing_score'] = df_with_scores.apply(calculate_processing_score, axis=1)
    df_with_scores['e_numbers'] = df_with_scores['ingredients_text'].map(count_e_numbers)
    
    metrics['total_products'] = len(_df)
    metrics['total_brands'] = _df['brand'].nunique()
    metrics['total_categories'] = _df['category_searched'].nunique()
    metrics['upf_percentage'] = (df_with_scores['processing_score'] > 60).sum() / len(df_with_scores) * 100
    metrics['avg_processing_score'] = df_with_scores['processing_score'].mean()
    metrics['avg_e_numbers'] = df_with_scores['e_numbers'].mean()
    metrics['high_sugar_pct'] = (_df['sugar_100g'] > 10).sum() / len(_df) * 100
    metrics['high_salt_pct'] = (_df['salt_100g'] > 1.5).sum() / len(_df) * 100
    
    return metrics

# --- MAIN APP ---
def main():
    # --- Handle alt_search_pending before any widgets ---
    if "alt_search_pending" in st.session_state:
        st.session_state["alt_search"] = st.session_state["alt_search_pending"]
        del st.session_state["alt_search_pending"]
    
    # --- Visitor counter (simple session-based) ---
    if "visit_count" not in st.session_state:
        st.session_state["visit_count"] = 1
    else:
        if "page_loaded" not in st.session_state:
            st.session_state["visit_count"] += 1
            st.session_state["page_loaded"] = True

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
        ["🏠 Overview", "🥗 Find Alternatives", "📊 Cluster Explorer", "🍬 Nutrition Insights", "🧪 Ingredients", "📈 Data", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    # Sidebar Footer
    st.sidebar.divider()
    st.sidebar.markdown("""
    <div class="footer-text">
        Built by <b>Sean Quinlan</b><br>
        <a href="https://linkedin.com/in/sean-quinlan-phd" target="_blank">🔗 LinkedIn</a> • 
        <a href="https://github.com/Synapsean" target="_blank">💻 GitHub</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Visitor counter in sidebar
    st.sidebar.markdown(f"""
    <div class="visitor-counter">
        👁️ Session views: {st.session_state.get('visit_count', 1)}
    </div>
    """, unsafe_allow_html=True)
    
    # Contact section
    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style="font-size: 0.9rem; color: #888; text-align: center;">
        <b>📬 Contact & Feedback</b><br>
        Found a bug or have a feature idea?
    </div>
    """, unsafe_allow_html=True)
    
    contact_col1, contact_col2 = st.sidebar.columns(2)
    with contact_col1:
        st.sidebar.markdown("""
        <a href="mailto:sean.quinlan91@gmail.com?subject=Food%20Detective%20Feedback" 
           style="text-decoration: none; color: #169B62;">
            📧 Email Me
        </a>
        """, unsafe_allow_html=True)
    with contact_col2:
        st.sidebar.markdown("""
        <a href="https://github.com/Synapsean/Irish-Food-Analysis/issues" 
           target="_blank" 
           style="text-decoration: none; color: #169B62;">
            🐛 Report Issue
        </a>
        """, unsafe_allow_html=True)
    
    # --- PAGE 0: OVERVIEW DASHBOARD ---
    if page == "🏠 Overview":
        st.markdown('<p class="main-header">🇮🇪 Irish Food Market Overview</p>', unsafe_allow_html=True)
        st.caption("Key insights into the Irish food market's processing levels and nutrition")
        
        with st.spinner("Calculating metrics..."):
            metrics = get_dashboard_metrics(df)
        
        # Top-level KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Products Analysed", 
                f"{metrics['total_products']:,}",
                help="Total products in our Irish food database"
            )
        with col2:
            st.metric(
                "Ultra-Processed Foods", 
                f"{metrics['upf_percentage']:.1f}%",
                delta=None,
                delta_color="inverse",
                help="Percentage of products with processing score > 60"
            )
        with col3:
            st.metric(
                "Avg Processing Score", 
                f"{metrics['avg_processing_score']:.0f}/100",
                help="Lower is better. 0=minimally processed, 100=ultra-processed"
            )
        with col4:
            st.metric(
                "Avg E-Numbers", 
                f"{metrics['avg_e_numbers']:.1f}",
                help="Average number of E-number additives per product"
            )
        
        st.divider()
        
        # Nutrition alerts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧂 High Salt Products")
            st.progress(metrics['high_salt_pct'] / 100)
            st.caption(f"{metrics['high_salt_pct']:.1f}% of products exceed 1.5g salt per 100g")
        with col2:
            st.markdown("### 🍬 High Sugar Products")
            st.progress(metrics['high_sugar_pct'] / 100)
            st.caption(f"{metrics['high_sugar_pct']:.1f}% of products exceed 10g sugar per 100g")
        
        st.divider()
        
        # Quick actions
        st.markdown("### 🚀 Get Started")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            st.markdown("""
            <div class="insight-box">
            <b>🥗 Find Healthier Options</b><br>
            Search for your favorite products and discover less-processed alternatives.
            </div>
            """, unsafe_allow_html=True)
        
        with action_col2:
            st.markdown("""
            <div class="insight-box">
            <b>📊 Explore Market Patterns</b><br>
            Use ML clustering to discover product groups with similar ingredients.
            </div>
            """, unsafe_allow_html=True)
        
        with action_col3:
            st.markdown("""
            <div class="insight-box">
            <b>🔬 Analyse Nutrition</b><br>
            Compare salt, sugar, and processing levels across food categories.
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Category breakdown
        st.markdown("### 📦 Products by Category")
        category_counts = df['category_searched'].value_counts().head(8)
        fig = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            labels={'x': 'Number of Products', 'y': 'Category'},
            color=category_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- PAGE 1: FIND ALTERNATIVES ---
    elif page == "🥗 Find Alternatives":
        st.markdown('<p class="main-header">🥗 Find Healthier Alternatives</p>', unsafe_allow_html=True)
        st.caption("Search for a product and discover less-processed options")
        
        if not RECOMMENDER_AVAILABLE:
            st.error("Recommender module not available. Check src/recommender.py")
            return
        
        # --- Handle example button clicks robustly ---
        def set_alt_search_pending(val):
            st.session_state["alt_search_pending"] = val

        # If a pending value is set, update alt_search and clear pending BEFORE widget is rendered
        if "alt_search_pending" in st.session_state:
            st.session_state["alt_search"] = st.session_state["alt_search_pending"]
            del st.session_state["alt_search_pending"]

        alt_search_val = st.session_state.get("alt_search", "")
        search_term = st.text_input(
            "🔍 Search for a product:", 
            value=alt_search_val,
            placeholder="e.g., Coca-Cola, Digestive Biscuits, Brennans Bread...",
            key="alt_search"
        )
        
        if search_term:
            # Find matching products
            matches = df[df['product_name'].str.lower().str.contains(search_term.lower(), na=False)]
            
            if matches.empty:
                st.warning(f"No products found matching '{search_term}'. Try a different search term.")
            else:
                # Let user select if multiple matches
                if len(matches) > 1:
                    product_options = matches['product_name'].tolist()[:10]
                    selected_name = st.selectbox(
                        f"Found {len(matches)} products. Select one:",
                        product_options
                    )
                    selected_product = matches[matches['product_name'] == selected_name].iloc[0].to_dict()
                else:
                    selected_product = matches.iloc[0].to_dict()
                    selected_name = selected_product['product_name']
                
                st.divider()
                
                # Analyse selected product
                analysis = get_product_analysis(selected_product)
                
                # Display product analysis
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📦 Your Product")
                    st.markdown(f"### {analysis['product_name']}")
                    st.caption(f"Brand: {analysis['brand']} | Category: {analysis['category']}")
                    
                    # Processing score gauge
                    score = analysis['processing_score']
                    badge, label = analysis['processing_badge'], analysis['processing_label']
                    
                    st.metric(
                        "Processing Score", 
                        f"{score}/100",
                        delta=None,
                        help="Score based on NOVA group (40%), E-number count (30%), and concerning additives (30%). Lower is healthier."
                    )
                    st.markdown(f"**{badge} {label}**")
                    
                    # Add explanation expander
                    with st.expander("ℹ️ What does this score mean?"):
                        st.markdown("""
                        **Processing Score Breakdown:**
                        - 🟢 **0-20**: Minimally processed (whole foods, basic preparation)
                        - 🟡 **21-40**: Lightly processed (simple ingredients added)
                        - 🟠 **41-60**: Moderately processed (several additives/processes)
                        - 🔴 **61-100**: Ultra-processed (many additives, highly industrial)
                        
                        The score considers:
                        - **NOVA classification** (WHO's food processing scale)
                        - **E-number count** (artificial additives)
                        - **Concerning ingredients** (sweeteners, preservatives, colors)
                        """)
                    
                    # Details
                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | NOVA Group | {analysis['nova_group']} ({analysis['nova_label']}) |
                    | E-Numbers | {analysis['e_numbers']} |
                    | Sugar | {analysis['sugar_100g']:.1f}g / 100g |
                    | Salt | {analysis['salt_100g']:.2f}g / 100g |
                    """)
                    
                    # Concerning additives
                    if analysis['concerning_additives']:
                        st.markdown("**⚠️ Additives Found:**")
                        for additive, desc in analysis['concerning_additives'][:5]:
                            st.caption(f"• {additive.title()}: {desc}")
                
                with col2:
                    st.subheader("🥗 Healthier Alternatives")
                    
                    # Find alternatives
                    alternatives = find_healthier_alternatives(selected_product, df, top_n=3)
                    
                    if not alternatives:
                        st.info("No healthier alternatives found in this category. This might already be one of the best options!")
                    else:
                        for i, alt in enumerate(alternatives, 1):
                            with st.container():
                                improvement = alt['score_improvement']
                                st.markdown(f"""
                                **{i}. {alt['product_name']}**  
                                {alt['processing_badge']} Score: {alt['processing_score']}/100 
                                <span style="color: green;">↓ {improvement} points better</span>
                                """, unsafe_allow_html=True)
                                
                                st.caption(f"NOVA: {alt['nova_group']} | E-numbers: {alt['e_numbers']} | Brand: {alt['brand']}")
                                st.divider()
                
                # Suggestions
                if analysis['suggestions']:
                    st.divider()
                    st.subheader("💡 Tips for Healthier Choices")
                    for suggestion in analysis['suggestions']:
                        st.markdown(f"• {suggestion}")
        
        else:
            # Show example products to try
            st.info("👆 Enter a product name above to get started!")
            
            # Show example with helpful context
            st.markdown("""
            ### 🔥 Try These Popular Searches
            Click any example below to see how it works:
            """)
            
            example_cols = st.columns(4)
            examples = ["Coca-Cola", "Digestive", "Soup", "Yogurt"]
            for col, example in zip(example_cols, examples):
                if col.button(f"🔍 {example}", use_container_width=True, key=f"example_{example}"):
                    set_alt_search_pending(example)
                    st.rerun()
            
            st.divider()
            
            # Add helpful stats
            st.markdown("### 💡 Did You Know?")
            tips_col1, tips_col2 = st.columns(2)
            with tips_col1:
                st.info("**67%** of Irish food products contain 3+ E-number additives")
            with tips_col2:
                st.info("**Ultra-processed foods** make up 40%+ of the analysed products")

    # --- PAGE 2: CLUSTER EXPLORER ---
    elif page == "📊 Cluster Explorer":
        st.markdown('<p class="main-header">🧠 Market Segmentation</p>', unsafe_allow_html=True)
        st.caption("K-Means clustering groups products by ingredient similarity")
        
        # Add help text
        with st.expander("ℹ️ How does clustering work?"):
            st.markdown("""
            This page uses **unsupervised machine learning** to automatically group similar products:
            
            1. **TF-IDF Vectorisation**: Converts ingredient lists into numerical features
            2. **K-Means Clustering**: Groups products with similar ingredient patterns
            3. **PCA Visualisation**: Projects high-dimensional data into 2D for visualisation
            
            Use the slider to experiment with different numbers of clusters (k). 
            Lower inertia = tighter, more cohesive clusters.
            """)
        
        n_clusters = st.sidebar.slider(
            "Clusters (k)", 2, 10, 5, 
            key="k_slider",
            help="Number of product groups to create. More clusters = more specific groupings."
        )
        
        with st.spinner("Running clustering..."):
            df_clustered, cluster_terms, inertia = run_clustering(df, n_clusters)
            # Add processing info columns
            df_clustered['processing_score'] = df_clustered.apply(calculate_processing_score, axis=1)
            df_clustered['badge'], df_clustered['proc_label'] = zip(*df_clustered['processing_score'].map(get_processing_badge))
            df_clustered['e_numbers'] = df_clustered['ingredients_text'].map(count_e_numbers)
            df_clustered['concerning_additives'] = df_clustered['ingredients_text'].map(find_concerning_additives)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.scatter(
                df_clustered, x='pca_x', y='pca_y', 
                color=df_clustered['cluster'].astype(str),
                hover_data=['product_name', 'category_searched', 'badge', 'proc_label', 'e_numbers'],
                title='Product Clusters (PCA Projection)',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=550, legend_title="Cluster")
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
            # Show cluster UPF stats
            st.markdown("#### Cluster Processing Overview")
            cluster_stats = df_clustered.groupby('cluster').agg(
                avg_score=('processing_score', 'mean'),
                upf_pct=('processing_score', lambda x: (x > 60).mean()*100),
                count=('product_name', 'count')
            )
            st.dataframe(cluster_stats, use_container_width=True)
        
        with col2:
            st.metric("Inertia", f"{inertia:,.0f}", help="Lower = tighter clusters")
            st.divider()
            for cid, terms in cluster_terms.items():
                count = (df_clustered['cluster'] == cid).sum()
                with st.expander(f"**Cluster {cid}** ({count} items)"):
                    st.write("🏷️ " + ", ".join(terms[:5]))
                    samples = df_clustered[df_clustered['cluster'] == cid][['product_name', 'badge', 'proc_label', 'e_numbers']].head(3)
                    for _, row in samples.iterrows():
                        st.caption(f"• {row['product_name']} {row['badge']} | E#: {row['e_numbers']}")
                    # Show why UPF for a sample UPF product
                    upf = df_clustered[(df_clustered['cluster'] == cid) & (df_clustered['processing_score'] > 60)]
                    if not upf.empty:
                        prod = upf.iloc[0]
                        with st.expander(f"⚠️ Why is this ultra-processed?"):
                            st.caption(f"**Example: {prod['product_name']}**")
                            if prod['concerning_additives']:
                                for add, desc in prod['concerning_additives'][:3]:
                                    st.caption(f"• {add.title()}: {desc}")
                            else:
                                st.caption("Contains multiple E-numbers or additives.")

    # --- PAGE 3: NUTRITION INSIGHTS ---
    elif page == "🍬 Nutrition Insights":
        st.markdown('<p class="main-header">🔬 Nutrition Analysis</p>', unsafe_allow_html=True)
        st.caption("Statistical analysis of salt and sugar content")
        
        # Add quick stats at top
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                "Avg Sugar", 
                f"{df['sugar_100g'].mean():.1f}g/100g",
                help="Average sugar content across all products"
            )
        with metric_col2:
            st.metric(
                "Avg Salt", 
                f"{df['salt_100g'].mean():.2f}g/100g",
                help="Average salt content across all products"
            )
        with metric_col3:
            st.metric(
                "Avg Fat", 
                f"{df['fat_100g'].mean():.1f}g/100g",
                help="Average fat content across all products"
            )
        
        st.divider()
        
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

    # --- PAGE 4: INGREDIENTS ---
    elif page == "🧪 Ingredients":
        st.markdown('<p class="main-header">🧪 Ingredient Analysis</p>', unsafe_allow_html=True)
        st.caption("Most common ingredients across Irish food products")
        
        with st.expander("ℹ️ Understanding ingredients"):
            st.markdown("""
            **Common Ingredient Types:**
            - 🌾 **Base ingredients**: Water, flour, sugar, salt
            - 🧪 **Additives**: E-numbers (preservatives, colors, flavors)
            - 🥛 **Derivatives**: Modified starches, vegetable oils, milk powder
            
            **E-Numbers to Watch:**
            - E621 (MSG), E330 (Citric acid), E407 (Carrageenan), E150 (Caramel color)
            """)
        
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
            matches = df[df['ingredients_text'].str.lower().str.contains(search.lower(), na=False)].copy()
            if not matches.empty:
                matches['processing_score'] = matches.apply(calculate_processing_score, axis=1)
                matches['badge'], matches['proc_label'] = zip(*matches['processing_score'].map(get_processing_badge))
                matches['e_numbers'] = matches['ingredients_text'].map(count_e_numbers)
                matches['concerning_additives'] = matches['ingredients_text'].map(find_concerning_additives)
            st.write(f"Found **{len(matches)}** products containing '{search}'")
            if not matches.empty:
                show_cols = ['product_name', 'brand', 'category_searched', 'badge', 'proc_label', 'e_numbers']
                st.dataframe(
                    matches[show_cols].head(20),
                    use_container_width=True, hide_index=True
                )
                # Show why UPF for first UPF product
                upf = matches[matches['processing_score'] > 60]
                if not upf.empty:
                    prod = upf.iloc[0]
                    st.markdown(f"**Why is '{prod['product_name']}' ultra-processed?**")
                    if prod['concerning_additives']:
                        for add, desc in prod['concerning_additives'][:3]:
                            st.caption(f"• {add.title()}: {desc}")
                    else:
                        st.caption("Contains multiple E-numbers or additives.")

    # --- PAGE 5: RAW DATA ---
    elif page == "📈 Data":
        st.markdown('<p class="main-header">📊 Data Explorer</p>', unsafe_allow_html=True)
        st.caption("Browse and filter the complete product database")
        
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
        # Add processing info columns
        if not filtered.empty:
            filtered['processing_score'] = filtered.apply(calculate_processing_score, axis=1)
            filtered['badge'], filtered['proc_label'] = zip(*filtered['processing_score'].map(get_processing_badge))
            filtered['e_numbers'] = filtered['ingredients_text'].map(count_e_numbers)
            filtered['concerning_additives'] = filtered['ingredients_text'].map(find_concerning_additives)
        
        st.write(f"Showing **{len(filtered)}** of {len(df)} products")
        
        if filtered.empty:
            st.warning("No products match your filters. Try adjusting your selection.")
        else:
            display_cols = ['product_name', 'brand', 'category_searched', 'sugar_100g', 'salt_100g', 'nova_group', 'badge', 'proc_label', 'e_numbers']
            
            # Format the dataframe for better presentation
            display_df = filtered[display_cols].copy()
            display_df = display_df.rename(columns={
                'product_name': 'Product',
                'brand': 'Brand',
                'category_searched': 'Category',
                'sugar_100g': 'Sugar (g/100g)',
                'salt_100g': 'Salt (g/100g)',
                'nova_group': 'NOVA',
                'badge': 'Status',
                'proc_label': 'Processing Level',
                'e_numbers': 'E-Numbers'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Sugar (g/100g)': st.column_config.NumberColumn(format="%.1f"),
                    'Salt (g/100g)': st.column_config.NumberColumn(format="%.2f"),
                    'Status': st.column_config.TextColumn(help="Processing status indicator"),
                }
            )
        # Show why UPF for first UPF product
        if not filtered.empty:
            upf = filtered[filtered['processing_score'] > 60]
            if not upf.empty:
                prod = upf.iloc[0]
                st.markdown(f"**Why is '{prod['product_name']}' ultra-processed?**")
                if prod['concerning_additives']:
                    for add, desc in prod['concerning_additives'][:3]:
                        st.caption(f"• {add.title()}: {desc}")
                else:
                    st.caption("Contains multiple E-numbers or additives.")
        # Download
        st.download_button(
            "📥 Download CSV", 
            filtered.to_csv(index=False), 
            "irish_food_data.csv", 
            "text/csv"
        )

    # --- PAGE 6: ABOUT ---
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