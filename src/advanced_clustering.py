"""
Advanced Clustering Analysis with Multiple Algorithms
===================================================

Implements and compares multiple clustering approaches:
- K-Means with various initialization strategies
- DBSCAN for density-based clustering  
- Hierarchical clustering with different linkage methods
- Gaussian Mixture Models for probabilistic clustering
- Advanced validation metrics and stability analysis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedClusterAnalyzer:
    """
    Comprehensive clustering analysis with multiple algorithms and validation.
    
    Provides:
    - Multiple clustering algorithms
    - Hyperparameter optimization
    - Clustering validation metrics
    - Stability analysis
    - Visualization tools
    """
    
    def __init__(self, products_df):
        self.df = products_df.copy()
        self.feature_matrix = None
        self.vectorizer = None
        self.scaler = None
        self.results = {}
        
    def prepare_features(self, feature_type='ingredients', tfidf_params=None):
        """
        Prepare feature matrix for clustering.
        
        Args:
            feature_type: 'ingredients', 'nutritional', or 'combined'
            tfidf_params: Parameters for TF-IDF vectorization
        """
        if feature_type == 'ingredients':
            self._prepare_ingredient_features(tfidf_params)
        elif feature_type == 'nutritional':
            self._prepare_nutritional_features()
        elif feature_type == 'combined':
            self._prepare_combined_features(tfidf_params)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def _prepare_ingredient_features(self, tfidf_params=None):
        """Prepare TF-IDF features from ingredients text."""
        default_params = {
            'max_features': 100,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.8
        }
        params = {**default_params, **(tfidf_params or {})}
        
        # Prepare ingredient texts
        ingredient_texts = []
        for _, row in self.df.iterrows():
            ingredients = row.get('ingredients_text', '')
            if pd.isna(ingredients) or ingredients == '':
                ingredients = 'unknown'
            ingredient_texts.append(str(ingredients).lower())
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(**params)
        self.feature_matrix = self.vectorizer.fit_transform(ingredient_texts).toarray()
        
        print(f"Ingredient features prepared: {self.feature_matrix.shape}")
        
    def _prepare_nutritional_features(self):
        """Prepare numerical features from nutritional data."""
        # Select numerical columns (customize based on your data)
        numerical_cols = ['energy_kcal', 'fat', 'saturated_fat', 'carbohydrates', 
                         'sugars', 'fiber', 'proteins', 'salt']
        
        # Filter available columns
        available_cols = [col for col in numerical_cols if col in self.df.columns]
        
        if not available_cols:
            raise ValueError("No nutritional columns found in data")
        
        # Prepare features
        feature_df = self.df[available_cols].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.mean())
        
        # Scale features
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(feature_df)
        
        print(f"Nutritional features prepared: {self.feature_matrix.shape}")
        
    def _prepare_combined_features(self, tfidf_params=None):
        """Combine ingredient and nutritional features."""
        # Get ingredient features
        self._prepare_ingredient_features(tfidf_params)
        ingredient_features = self.feature_matrix.copy()
        
        # Get nutritional features
        try:
            self._prepare_nutritional_features()
            nutritional_features = self.feature_matrix.copy()
            
            # Combine features
            self.feature_matrix = np.hstack([ingredient_features, nutritional_features])
            print(f"Combined features prepared: {self.feature_matrix.shape}")
            
        except ValueError:
            # Fall back to ingredient features only
            print("Nutritional features not available, using ingredients only")
            self.feature_matrix = ingredient_features
    
    def kmeans_analysis(self, k_range=(2, 15), n_init_strategies=5):
        """
        Comprehensive K-Means analysis with multiple initialization strategies.
        
        Args:
            k_range: Range of k values to test
            n_init_strategies: Number of different random initializations
        
        Returns:
            Dict with results for each k value
        """
        results = {}
        
        # Test different k values
        for k in range(k_range[0], k_range[1] + 1):
            k_results = []
            
            # Multiple runs with different initializations
            for run in range(n_init_strategies):
                kmeans = KMeans(
                    n_clusters=k,
                    n_init=10,
                    max_iter=300,
                    random_state=run,
                    algorithm='lloyd'
                )
                
                cluster_labels = kmeans.fit_predict(self.feature_matrix)
                
                # Calculate validation metrics
                silhouette = silhouette_score(self.feature_matrix, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(self.feature_matrix, cluster_labels)
                inertia = kmeans.inertia_
                
                k_results.append({
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski_harabasz,
                    'inertia': inertia,
                    'cluster_labels': cluster_labels,
                    'model': kmeans
                })
            
            # Find best run for this k
            best_run = max(k_results, key=lambda x: x['silhouette_score'])
            
            results[k] = {
                'best_silhouette': best_run['silhouette_score'],
                'best_calinski_harabasz': best_run['calinski_harabasz_score'],
                'best_inertia': best_run['inertia'],
                'best_labels': best_run['cluster_labels'],
                'best_model': best_run['model'],
                'stability': np.std([r['silhouette_score'] for r in k_results]),
                'all_runs': k_results
            }
        
        self.results['kmeans'] = results
        return results
    
    def dbscan_analysis(self, eps_range=None, min_samples_range=None):
        """
        DBSCAN clustering with parameter grid search.
        
        DBSCAN advantages:
        - Finds clusters of arbitrary shape
        - Automatically determines number of clusters
        - Identifies outliers
        """
        if eps_range is None:
            eps_range = np.arange(0.1, 2.0, 0.1)
        if min_samples_range is None:
            min_samples_range = range(3, 20)
        
        results = {}
        best_score = -1
        best_params = None
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(self.feature_matrix)
                
                # Skip if all points are noise or only one cluster
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                if n_clusters < 2:
                    continue
                
                # Calculate metrics
                silhouette = silhouette_score(self.feature_matrix, cluster_labels)
                n_noise = list(cluster_labels).count(-1)
                noise_ratio = n_noise / len(cluster_labels)
                
                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'n_noise_points': n_noise,
                    'noise_ratio': noise_ratio,
                    'cluster_labels': cluster_labels
                }
                
                results[f"eps_{eps:.1f}_min_{min_samples}"] = result
                
                if silhouette > best_score:
                    best_score = silhouette
                    best_params = result
        
        self.results['dbscan'] = {
            'best_params': best_params,
            'all_results': results
        }
        return self.results['dbscan']
    
    def hierarchical_analysis(self, n_clusters_range=(2, 15), linkage_methods=None):
        """
        Hierarchical clustering with different linkage methods.
        
        Linkage methods:
        - ward: Minimizes variance within clusters
        - complete: Maximum distance between cluster elements
        - average: Average distance between cluster elements
        - single: Minimum distance between cluster elements
        """
        if linkage_methods is None:
            linkage_methods = ['ward', 'complete', 'average', 'single']
        
        results = {}
        
        for linkage in linkage_methods:
            linkage_results = {}
            
            for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
                hierarchical = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage
                )
                
                cluster_labels = hierarchical.fit_predict(self.feature_matrix)
                
                # Calculate metrics
                silhouette = silhouette_score(self.feature_matrix, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(self.feature_matrix, cluster_labels)
                
                linkage_results[n_clusters] = {
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski_harabasz,
                    'cluster_labels': cluster_labels
                }
            
            results[linkage] = linkage_results
        
        self.results['hierarchical'] = results
        return results
    
    def gaussian_mixture_analysis(self, n_components_range=(2, 15), covariance_types=None):
        """
        Gaussian Mixture Model clustering.
        
        Provides probabilistic cluster assignments and can handle:
        - Overlapping clusters
        - Different cluster shapes (with different covariance types)
        """
        if covariance_types is None:
            covariance_types = ['full', 'tied', 'diag', 'spherical']
        
        results = {}
        
        for cov_type in covariance_types:
            cov_results = {}
            
            for n_components in range(n_components_range[0], n_components_range[1] + 1):
                try:
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=cov_type,
                        random_state=42
                    )
                    
                    cluster_labels = gmm.fit_predict(self.feature_matrix)
                    cluster_probs = gmm.predict_proba(self.feature_matrix)
                    
                    # Calculate metrics
                    silhouette = silhouette_score(self.feature_matrix, cluster_labels)
                    bic = gmm.bic(self.feature_matrix)
                    aic = gmm.aic(self.feature_matrix)
                    log_likelihood = gmm.score(self.feature_matrix)
                    
                    cov_results[n_components] = {
                        'silhouette_score': silhouette,
                        'bic': bic,
                        'aic': aic,
                        'log_likelihood': log_likelihood,
                        'cluster_labels': cluster_labels,
                        'cluster_probabilities': cluster_probs,
                        'model': gmm
                    }
                    
                except Exception as e:
                    print(f"Error with GMM {cov_type} {n_components} components: {e}")
            
            results[cov_type] = cov_results
        
        self.results['gaussian_mixture'] = results
        return results
    
    def clustering_stability_analysis(self, algorithm='kmeans', n_bootstrap=10, subsample_ratio=0.8):
        """
        Analyze clustering stability using bootstrap sampling.
        
        Stability measures:
        - How consistent are cluster assignments across different samples?
        - Adjusted Rand Index between different runs
        """
        n_samples = int(len(self.feature_matrix) * subsample_ratio)
        stability_scores = []
        
        # Base clustering on full data
        if algorithm == 'kmeans':
            base_model = KMeans(n_clusters=5, random_state=42)  # Use optimal k
        elif algorithm == 'dbscan':
            base_model = DBSCAN(eps=0.5, min_samples=5)  # Use optimal params
        else:
            raise ValueError(f"Stability analysis not implemented for {algorithm}")
        
        base_labels = base_model.fit_predict(self.feature_matrix)
        
        # Bootstrap sampling
        for i in range(n_bootstrap):
            # Sample subset
            indices = np.random.choice(len(self.feature_matrix), n_samples, replace=False)
            subset_features = self.feature_matrix[indices]
            
            # Cluster subset
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=5, random_state=i)
            elif algorithm == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
            
            subset_labels = model.fit_predict(subset_features)
            
            # Compare with base clustering (on same indices)
            base_subset_labels = base_labels[indices]
            
            # Calculate adjusted rand index
            ari = adjusted_rand_score(base_subset_labels, subset_labels)
            stability_scores.append(ari)
        
        stability_result = {
            'mean_ari': np.mean(stability_scores),
            'std_ari': np.std(stability_scores),
            'all_scores': stability_scores
        }
        
        self.results[f'{algorithm}_stability'] = stability_result
        return stability_result
    
    def cluster_interpretation(self, cluster_labels, algorithm_name, top_n_features=10):
        """
        Interpret clusters by finding characteristic features.
        
        For ingredient-based clustering:
        - Find most common ingredients per cluster
        - Calculate TF-IDF weights per cluster
        """
        interpretation = {}
        n_clusters = len(set(cluster_labels))
        
        if self.vectorizer is not None:  # Ingredient-based features
            feature_names = self.vectorizer.get_feature_names_out()
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_features = self.feature_matrix[cluster_mask]
                
                # Average TF-IDF scores for this cluster
                avg_scores = np.mean(cluster_features, axis=0)
                
                # Top features
                top_indices = np.argsort(avg_scores)[-top_n_features:][::-1]
                top_features = [(feature_names[i], avg_scores[i]) for i in top_indices]
                
                # Sample products in cluster
                cluster_products = self.df[cluster_mask]['product_name'].head(5).tolist()
                
                interpretation[f'cluster_{cluster_id}'] = {
                    'top_ingredients': top_features,
                    'n_products': np.sum(cluster_mask),
                    'sample_products': cluster_products
                }
        
        self.results[f'{algorithm_name}_interpretation'] = interpretation
        return interpretation
    
    def visualize_clusters(self, cluster_labels, algorithm_name, save_path=None):
        """
        Create PCA visualization of clusters.
        """
        # PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.feature_matrix)
        
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = set(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points in DBSCAN
                color = 'black'
                marker = 'x'
                alpha = 0.5
            else:
                marker = 'o'
                alpha = 0.7
            
            mask = cluster_labels == label
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1],
                c=[color], 
                marker=marker,
                alpha=alpha,
                label=f'Cluster {label}' if label != -1 else 'Noise'
            )
        
        plt.title(f'{algorithm_name} Clustering Results (PCA Visualization)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def comprehensive_clustering_report(self):
        """
        Generate comprehensive clustering comparison report.
        """
        print("=== COMPREHENSIVE CLUSTERING ANALYSIS REPORT ===\\n")
        
        # Run all clustering algorithms
        print("1. Running K-Means analysis...")
        kmeans_results = self.kmeans_analysis()
        
        print("2. Running DBSCAN analysis...")
        dbscan_results = self.dbscan_analysis()
        
        print("3. Running Hierarchical clustering...")
        hierarchical_results = self.hierarchical_analysis()
        
        print("4. Running Gaussian Mixture Models...")
        gmm_results = self.gaussian_mixture_analysis()
        
        # Find best configurations
        print("\\n=== BEST CONFIGURATIONS ===")
        
        # Best K-Means
        best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['best_silhouette'])
        print(f"K-Means: k={best_k}, Silhouette={kmeans_results[best_k]['best_silhouette']:.4f}")
        
        # Best DBSCAN
        if dbscan_results['best_params']:
            dbscan_best = dbscan_results['best_params']
            print(f"DBSCAN: eps={dbscan_best['eps']:.2f}, min_samples={dbscan_best['min_samples']}, "
                  f"Silhouette={dbscan_best['silhouette_score']:.4f}")
        
        # Stability analysis
        print("\\n5. Running stability analysis...")
        kmeans_stability = self.clustering_stability_analysis('kmeans')
        print(f"K-Means stability (ARI): {kmeans_stability['mean_ari']:.4f} ± {kmeans_stability['std_ari']:.4f}")
        
        # Interpretation
        best_kmeans_labels = kmeans_results[best_k]['best_labels']
        interpretation = self.cluster_interpretation(best_kmeans_labels, 'kmeans')
        
        print(f"\\n=== CLUSTER INTERPRETATION (K-Means, k={best_k}) ===")
        for cluster_name, info in interpretation.items():
            print(f"\\n{cluster_name.upper()} ({info['n_products']} products):")
            print("Top ingredients:")
            for ingredient, score in info['top_ingredients'][:5]:
                print(f"  - {ingredient}: {score:.4f}")
        
        return {
            'kmeans': kmeans_results,
            'dbscan': dbscan_results,
            'hierarchical': hierarchical_results,
            'gaussian_mixture': gmm_results,
            'stability': {
                'kmeans': kmeans_stability
            }
        }


if __name__ == "__main__":
    print("Advanced Clustering Analysis")
    print("Note: Update with your actual data loading logic")
    
    # Example usage:
    # df = load_your_data()
    # analyzer = AdvancedClusterAnalyzer(df)
    # analyzer.prepare_features('ingredients')
    # results = analyzer.comprehensive_clustering_report()