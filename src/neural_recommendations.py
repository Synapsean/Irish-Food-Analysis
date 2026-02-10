"""
Neural Ingredient Embeddings for Food Recommendation
==================================================

Implements neural network-based recommendation system:
- Word2Vec-style embeddings for ingredients
- Neural collaborative filtering
- Deep learning recommendation with PyTorch
- Embedding visualization with t-SNE/UMAP
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional deep learning dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim not available. Install with: pip install gensim")

try:
    from sklearn.manifold import TSNE
    import umap.umap_ as umap
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False
    print("Manifold learning libs not available. Install with: pip install umap-learn")


class IngredientEmbeddings:
    """
    Learn dense vector representations of ingredients using Word2Vec-style training.
    
    Similar ingredients (e.g., "sugar", "glucose syrup") will have similar embeddings.
    """
    
    def __init__(self, embedding_dim=100, window_size=5, min_count=2):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.model = None
        self.ingredient_vocab = None
        
    def prepare_ingredient_sequences(self, products_df):
        """
        Convert ingredient lists into sequences for training.
        
        Each product becomes a "sentence" of ingredients.
        """
        sequences = []
        
        for _, row in products_df.iterrows():
            ingredients = row.get('ingredients_text', '')
            if pd.isna(ingredients) or ingredients == '':
                continue
                
            # Simple tokenization (you might want to use the tokenizer module)
            ingredient_list = [
                ingredient.strip().lower() 
                for ingredient in str(ingredients).split(',')
                if ingredient.strip()
            ]
            
            if len(ingredient_list) >= 2:  # Need at least 2 ingredients
                sequences.append(ingredient_list)
        
        print(f"Prepared {len(sequences)} ingredient sequences")
        return sequences
    
    def train_embeddings(self, sequences):
        """
        Train Word2Vec embeddings on ingredient sequences.
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim required for embeddings")
        
        self.model = Word2Vec(
            sentences=sequences,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=self.min_count,
            workers=4,
            epochs=100,
            seed=42
        )
        
        self.ingredient_vocab = set(self.model.wv.index_to_key)
        print(f"Trained embeddings for {len(self.ingredient_vocab)} ingredients")
        
    def get_embedding(self, ingredient):
        """Get embedding vector for an ingredient."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        ingredient = ingredient.lower().strip()
        if ingredient in self.model.wv:
            return self.model.wv[ingredient]
        else:
            return np.zeros(self.embedding_dim)  # OOV handling
    
    def find_similar_ingredients(self, ingredient, top_n=10):
        """Find ingredients similar to the given ingredient."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        ingredient = ingredient.lower().strip()
        if ingredient not in self.model.wv:
            return []
        
        similar = self.model.wv.most_similar(ingredient, topn=top_n)
        return similar
    
    def ingredient_analogy(self, positive, negative, top_n=5):
        """
        Ingredient analogies like "sugar is to sweet as salt is to ?"
        
        Args:
            positive: List of positive ingredients
            negative: List of negative ingredients
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        try:
            results = self.model.wv.most_similar(
                positive=positive, 
                negative=negative, 
                topn=top_n
            )
            return results
        except KeyError as e:
            print(f"Ingredient not in vocabulary: {e}")
            return []


class NeuralRecommendationSystem(nn.Module):
    """
    Deep learning recommendation system using PyTorch.
    
    Architecture:
    - Embedding layers for products and users (implicit feedback)
    - Deep neural network for learning complex interactions
    - Multiple loss functions (BCE, MSE, ranking loss)
    """
    
    def __init__(self, n_products, n_categories, embedding_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for neural recommendations")
        
        self.n_products = n_products
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.product_embedding = nn.Embedding(n_products, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim // 2)
        
        # Deep network
        layers = []
        input_dim = embedding_dim + embedding_dim // 2  # product + category
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.normal_(self.product_embedding.weight, std=0.1)
        nn.init.normal_(self.category_embedding.weight, std=0.1)
    
    def forward(self, product_ids, category_ids):
        """
        Forward pass.
        
        Args:
            product_ids: Tensor of product indices
            category_ids: Tensor of category indices
        """
        # Get embeddings
        product_emb = self.product_embedding(product_ids)
        category_emb = self.category_embedding(category_ids)
        
        # Concatenate features
        features = torch.cat([product_emb, category_emb], dim=1)
        
        # Deep network
        output = self.network(features)
        return output.squeeze()


class ProductInteractionDataset(Dataset):
    """
    Dataset for training neural recommendation system.
    
    Creates positive and negative examples:
    - Positive: Products in same category (similar)
    - Negative: Products in different categories (dissimilar)
    """
    
    def __init__(self, products_df, negative_sampling_ratio=1.0):
        self.df = products_df.copy()
        self.product_to_idx = {name: idx for idx, name in enumerate(self.df['product_name'])}
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.df['main_category'].unique())}
        
        self.positive_pairs = self._create_positive_pairs()
        self.negative_pairs = self._create_negative_pairs(negative_sampling_ratio)
        
        # Combine all pairs
        self.all_pairs = (
            [(p1, p2, 1) for p1, p2 in self.positive_pairs] +  # Similar pairs (label=1)
            [(p1, p2, 0) for p1, p2 in self.negative_pairs]    # Dissimilar pairs (label=0)
        )
        
        print(f"Dataset created: {len(self.positive_pairs)} positive, {len(self.negative_pairs)} negative pairs")
    
    def _create_positive_pairs(self):
        """Create pairs of products in same category."""
        positive_pairs = []
        
        for category in self.df['main_category'].unique():
            category_products = self.df[self.df['main_category'] == category]['product_name'].tolist()
            
            # Create all pairs within category
            for i in range(len(category_products)):
                for j in range(i + 1, len(category_products)):
                    positive_pairs.append((category_products[i], category_products[j]))
        
        return positive_pairs
    
    def _create_negative_pairs(self, ratio):
        """Create pairs of products from different categories."""
        negative_pairs = []
        n_negative = int(len(self.positive_pairs) * ratio)
        
        categories = self.df['main_category'].unique()
        
        for _ in range(n_negative):
            # Sample two different categories
            cat1, cat2 = np.random.choice(categories, 2, replace=False)
            
            # Sample product from each category
            prod1 = np.random.choice(
                self.df[self.df['main_category'] == cat1]['product_name'].tolist()
            )
            prod2 = np.random.choice(
                self.df[self.df['main_category'] == cat2]['product_name'].tolist()
            )
            
            negative_pairs.append((prod1, prod2))
        
        return negative_pairs
    
    def __len__(self):
        return len(self.all_pairs)
    
    def __getitem__(self, idx):
        product1_name, product2_name, label = self.all_pairs[idx]
        
        # Get indices and categories
        product1_idx = self.product_to_idx[product1_name]
        product2_idx = self.product_to_idx[product2_name]
        
        product1_category = self.df[self.df['product_name'] == product1_name]['main_category'].iloc[0]
        product2_category = self.df[self.df['product_name'] == product2_name]['main_category'].iloc[0]
        
        category1_idx = self.category_to_idx[product1_category]
        category2_idx = self.category_to_idx[product2_category]
        
        return {
            'product1_id': torch.tensor(product1_idx, dtype=torch.long),
            'product2_id': torch.tensor(product2_idx, dtype=torch.long),
            'category1_id': torch.tensor(category1_idx, dtype=torch.long),
            'category2_id': torch.tensor(category2_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


class NeuralRecommendationTrainer:
    """
    Training pipeline for neural recommendation system.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Move to device
            product1_ids = batch['product1_id'].to(self.device)
            category1_ids = batch['category1_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass (using first product)
            predictions = self.model(product1_ids, category1_ids)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                product1_ids = batch['product1_id'].to(self.device)
                category1_ids = batch['category1_id'].to(self.device)
                labels = batch['label'].to(self.device)
                
                predictions = self.model(product1_ids, category1_ids)
                
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted_labels = (predictions > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_neural_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if patience_counter >= 10:
                print("Early stopping triggered")
                break


def create_neural_recommendation_pipeline(products_df):
    """
    Complete pipeline for training neural recommendation system.
    """
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        return None
    
    print("Creating neural recommendation pipeline...")
    
    # Create dataset
    dataset = ProductInteractionDataset(products_df)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model
    n_products = len(dataset.product_to_idx)
    n_categories = len(dataset.category_to_idx)
    
    model = NeuralRecommendationSystem(
        n_products=n_products,
        n_categories=n_categories,
        embedding_dim=64,
        hidden_dims=[128, 64, 32]
    )
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = NeuralRecommendationTrainer(model, device)
    
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs=100)
    
    return {
        'model': model,
        'dataset': dataset,
        'trainer': trainer,
        'device': device
    }


def visualize_embeddings(embedding_model, ingredients_to_plot=None, method='tsne'):
    """
    Visualize ingredient embeddings in 2D using t-SNE or UMAP.
    """
    if not MANIFOLD_AVAILABLE:
        print("Manifold learning libraries not available")
        return
    
    if embedding_model.model is None:
        print("Embeddings not trained yet")
        return
    
    # Get embeddings
    if ingredients_to_plot is None:
        # Use most common ingredients
        ingredients_to_plot = list(embedding_model.ingredient_vocab)[:100]
    
    embeddings = []
    labels = []
    
    for ingredient in ingredients_to_plot:
        if ingredient in embedding_model.model.wv:
            embeddings.append(embedding_model.model.wv[ingredient])
            labels.append(ingredient)
    
    embeddings = np.array(embeddings)
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    
    # Annotate points
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    plt.title(f'Ingredient Embeddings Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Neural Ingredient Embeddings and Recommendation System")
    print("This module requires: pip install torch gensim umap-learn")
    
    # Example usage:
    # df = load_your_data()
    # 
    # # Train ingredient embeddings
    # embeddings = IngredientEmbeddings()
    # sequences = embeddings.prepare_ingredient_sequences(df)
    # embeddings.train_embeddings(sequences)
    # 
    # # Train neural recommendation system
    # neural_pipeline = create_neural_recommendation_pipeline(df)
    # 
    # # Visualize embeddings
    # visualize_embeddings(embeddings)