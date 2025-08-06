"""
CS Paper Recommender - Similarity Engine
========================================

Today we're implementing the CORE of recommendation systems:
- Cosine Similarity (how Netflix finds similar movies)
- TF-IDF Vectorization (turning text into math)
- Content-based recommendations (your first working recommender!)

This is where features become predictions!
"""

import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

class PaperSimilarityEngine:
    def __init__(self, papers_file='data/raw/cs_papers.json'):
        """
        Initialize our similarity engine.
        
        ML Concept: We're building a system that can mathematically 
        compute "how similar are these two papers?"
        """
        print("🚀 Loading papers and building similarity engine...")
        
        # Load our scraped data
        with open(papers_file, 'r', encoding='utf-8') as f:
            self.papers_data = json.load(f)
        
        self.papers_df = pd.DataFrame(self.papers_data)
        print(f"📊 Loaded {len(self.papers_df)} papers")
        
        # Initialize our ML components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
        print("✅ Similarity engine initialized!")
    
    def preprocess_text_for_ml(self, papers_df):
        """
        CORE ML CONCEPT: Text Preprocessing for Similarity
        ================================================
        
        Raw text: "Deep Learning for Computer Vision Applications"
        Processed: "deep learn comput vision applic"
        
        Why? We want to focus on MEANING, not exact word matching.
        """
        print("🧹 Preprocessing text for ML algorithms...")
        
        processed_papers = []
        
        for _, paper in papers_df.iterrows():
            # Combine title, abstract, and keywords for rich content representation
            content_parts = []
            
            # Title gets extra weight (appears 2x)
            if paper.get('clean_title'):
                content_parts.extend([paper['clean_title']] * 2)
            
            # Abstract content
            if paper.get('clean_abstract'):
                content_parts.append(paper['clean_abstract'])
            
            # Keywords get extra weight (they're curated!)
            if paper.get('keywords'):
                keyword_text = ' '.join(paper['keywords'])
                content_parts.extend([keyword_text] * 3)
            
            # Categories (convert to readable text)
            if paper.get('primary_category'):
                # Convert 'cs.AI' to 'artificial intelligence'
                category_text = self.category_to_text(paper['primary_category'])
                content_parts.append(category_text)
            
            # Combine all content
            full_content = ' '.join(content_parts)
            processed_papers.append(full_content)
        
        print(f"✅ Processed {len(processed_papers)} papers for similarity computation")
        return processed_papers
    
    def category_to_text(self, category):
        """
        Convert category codes to meaningful text for better similarity
        """
        category_map = {
            'cs.AI': 'artificial intelligence machine learning',
            'cs.LG': 'machine learning statistical learning',
            'cs.CV': 'computer vision image processing',
            'cs.CL': 'natural language processing computational linguistics',
            'cs.CR': 'cryptography security privacy',
            'cs.DB': 'database systems data management',
            'cs.DC': 'distributed computing parallel systems',
            'cs.DS': 'data structures algorithms',
            'cs.HC': 'human computer interaction usability',
            'cs.IR': 'information retrieval search systems',
            'cs.NI': 'networking protocols internet',
            'cs.OS': 'operating systems system software',
            'cs.PL': 'programming languages compilers',
            'cs.SE': 'software engineering development',
            'cs.SY': 'systems control automation'
        }
        return category_map.get(category, category)
    
    def build_tfidf_vectors(self):
        """
        CORE ML CONCEPT: TF-IDF Vectorization
        ====================================
        
        This converts text into numbers that capture semantic meaning!
        
        TF-IDF = Term Frequency × Inverse Document Frequency
        
        Example:
        - "neural" appears 5 times in paper A, 1000 papers total have "neural"
        - "transformer" appears 3 times in paper A, 50 papers total have "transformer"  
        - "transformer" gets HIGHER weight (more distinctive!)
        
        This is how we capture what makes each paper unique!
        """
        print("🔢 Building TF-IDF vectors...")
        
        # Preprocess all papers
        processed_texts = self.preprocess_text_for_ml(self.papers_df)
        
        # Initialize TF-IDF vectorizer with smart parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,      # Keep top 5000 most important words
            min_df=2,               # Word must appear in at least 2 papers
            max_df=0.8,             # Ignore words in >80% of papers (too common)
            stop_words='english',   # Remove "the", "and", etc.
            ngram_range=(1, 2),     # Include both single words and pairs
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only letters, 2+ chars
        )
        
        # Transform text to TF-IDF vectors
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        print(f"✅ Created TF-IDF matrix: {self.tfidf_matrix.shape}")
        print(f"   📊 {self.tfidf_matrix.shape[0]} papers × {self.tfidf_matrix.shape[1]} features")
        
        # Show some example features
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"   🔤 Sample features: {list(feature_names[:10])}")
        
        return self.tfidf_matrix
    
    def compute_similarity_matrix(self):
        """
        CORE ML CONCEPT: Cosine Similarity
        =================================
        
        This is THE algorithm behind most recommendation systems!
        
        Cosine similarity measures the angle between two vectors:
        - Same direction (similar papers) = score close to 1
        - Opposite direction (different papers) = score close to 0
        - Perpendicular (unrelated papers) = score of 0.5
        
        Why cosine? It ignores paper length, focuses on content similarity.
        """
        print("🧮 Computing cosine similarity matrix...")
        
        if self.tfidf_matrix is None:
            self.build_tfidf_vectors()
        
        # Compute pairwise cosine similarity
        # This creates a matrix where entry (i,j) = similarity between paper i and paper j
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print(f"✅ Computed similarity matrix: {self.similarity_matrix.shape}")
        print(f"   📊 {self.similarity_matrix.shape[0]} × {self.similarity_matrix.shape[1]} similarity scores")
        
        # Show some statistics
        similarities = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        print(f"   📈 Similarity stats:")
        print(f"      Mean: {similarities.mean():.3f}")
        print(f"      Max:  {similarities.max():.3f}")
        print(f"      Min:  {similarities.min():.3f}")
        
        return self.similarity_matrix
    
    def find_similar_papers(self, paper_index, top_k=5):
        """
        CORE RECOMMENDER FUNCTION
        ========================
        
        This is it! Given a paper, find the most similar papers.
        This is exactly what Netflix does: "People who watched X also watched Y"
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # Get similarity scores for this paper with all others
        paper_similarities = self.similarity_matrix[paper_index]
        
        # Get indices of most similar papers (excluding the paper itself)
        similar_indices = np.argsort(paper_similarities)[::-1][1:top_k+1]
        
        # Return papers with their similarity scores
        recommendations = []
        for idx in similar_indices:
            paper = self.papers_df.iloc[idx]
            similarity_score = paper_similarities[idx]
            
            recommendations.append({
                'paper_index': idx,
                'title': paper['title'],
                'similarity_score': similarity_score,
                'authors': paper['authors'][:3],  # First 3 authors
                'primary_category': paper['primary_category'],
                'keywords': paper['keywords'][:5],  # Top 5 keywords
                'url': paper['url']
            })
        
        return recommendations
    
    def recommend_by_title(self, search_title, top_k=5):
        """
        User-friendly recommendation function
        =====================================
        
        Let users search by paper title and get recommendations!
        """
        # Find papers with similar titles
        matches = self.papers_df[
            self.papers_df['title'].str.lower().str.contains(
                search_title.lower(), na=False
            )
        ]
        
        if matches.empty:
            print(f"❌ No papers found matching '{search_title}'")
            return []
        
        # Use the first match
        paper_index = matches.index[0]
        paper = matches.iloc[0]
        
        print(f"🎯 Finding papers similar to:")
        print(f"   📄 '{paper['title']}'")
        print(f"   👥 Authors: {', '.join(paper['authors'][:3])}")
        print(f"   🏷️  Category: {paper['primary_category']}")
        
        # Get recommendations
        recommendations = self.find_similar_papers(paper_index, top_k)
        
        return recommendations
    
    def display_recommendations(self, recommendations):
        """
        Pretty print recommendations for human review
        """
        print(f"\n🔥 TOP RECOMMENDATIONS:")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. 📄 {rec['title']}")
            print(f"   🎯 Similarity: {rec['similarity_score']:.3f}")
            print(f"   👥 Authors: {', '.join(rec['authors'])}")
            print(f"   🏷️  Category: {rec['primary_category']}")
            print(f"   🔑 Keywords: {', '.join(rec['keywords'])}")
            print(f"   🔗 URL: {rec['url']}")
    
    def analyze_similarity_quality(self, sample_size=5):
        """
        ML Engineering: Evaluate our similarity algorithm
        ================================================
        
        Before deploying, we need to check if our similarities make sense!
        """
        print("\n🔍 SIMILARITY QUALITY ANALYSIS")
        print("=" * 50)
        
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # Sample random papers and show their most similar papers
        sample_indices = np.random.choice(len(self.papers_df), sample_size, replace=False)
        
        for idx in sample_indices:
            paper = self.papers_df.iloc[idx]
            recommendations = self.find_similar_papers(idx, top_k=3)
            
            print(f"\n📄 BASE PAPER: {paper['title'][:60]}...")
            print(f"🏷️  Category: {paper['primary_category']}")
            print(f"🔑 Keywords: {', '.join(paper['keywords'][:3])}")
            
            print("   🎯 Most Similar:")
            for i, rec in enumerate(recommendations, 1):
                print(f"      {i}. {rec['title'][:50]}... (sim: {rec['similarity_score']:.3f})")
            
            print("-" * 50)

# Let's test our similarity engine!
if __name__ == "__main__":
    print("🚀 Welcome to Paper Similarity Engine!")
    print("Building your first content-based recommender...\n")
    
    # Initialize engine
    engine = PaperSimilarityEngine()
    
    # Build the ML pipeline
    engine.build_tfidf_vectors()
    engine.compute_similarity_matrix()
    
    # Test with some recommendations
    print("\n" + "="*80)
    print("🎯 TESTING RECOMMENDATIONS")
    print("="*80)
    
    # Example 1: Find papers similar to something about neural networks
    print("\n🔍 DEMO 1: Search for neural network papers")
    recommendations = engine.recommend_by_title("neural", top_k=3)
    engine.display_recommendations(recommendations)
    
    # Example 2: Analyze similarity quality
    engine.analyze_similarity_quality(sample_size=3)
    
    print("\n✨ Recommendation engine complete!")
    print("🎉 You just built the same algorithms that power Netflix recommendations!")