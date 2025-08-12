"""
CS Paper Recommender - ENHANCED Hybrid Similarity Engine

Hybrid:
- TF-IDF (keyword matching) 
- Sentence Transformers (semantic meaning)
- Author similarity (research network effects)

"""

import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer #new library for "deep learning" <-> "neural network"
from collections import Counter

class EnhancedPaperSimilarityEngine:
    def __init__(self, papers_file='data/raw/cs_papers.json'):

        print("🚀 Loading Enhanced Similarity Engine...")
        
        # Load data
        with open(papers_file, 'r', encoding='utf-8') as f:
            self.papers_data = json.load(f)
        
        self.papers_df = pd.DataFrame(self.papers_data) #convert python dict to pandas data frame
        print(f"📊 Loaded {len(self.papers_df)} papers")
        
        # TF-IDF components 
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_similarity_matrix = None
        
        # NEW: Sentence Transformer components
        print("🧠 Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
        self.sentence_embeddings = None
        self.semantic_similarity_matrix = None
        
        # NEW: Author similarity components  
        self.author_similarity_matrix = None
        
        # Final hybrid similarity matrix
        self.hybrid_similarity_matrix = None
        
        print("✅ Enhanced engine initialized!")
    
    def preprocess_text_for_ml(self, papers_df):
        
        print("🧹 Preprocessing text for ML algorithms...")
        
        processed_papers = []
        
        for _, paper in papers_df.iterrows():
            content_parts = []
            
            if paper.get('clean_title'):
                content_parts.extend([paper['clean_title']] * 2)
            
            if paper.get('clean_abstract'):
                content_parts.append(paper['clean_abstract'])
            
            if paper.get('keywords'):
                keyword_text = ' '.join(paper['keywords'])
                content_parts.extend([keyword_text] * 3)
            
            if paper.get('primary_category'):
                category_text = self.category_to_text(paper['primary_category'])
                content_parts.append(category_text)
            
            full_content = ' '.join(content_parts)
            processed_papers.append(full_content) #list which has one item per paper
        
        print(f"✅ Processed {len(processed_papers)} papers")
        return processed_papers
    
    def category_to_text(self, category):
        """SAME category mapping as before"""
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
        
        print("🔢 Building TF-IDF vectors...")
        
        processed_texts = self.preprocess_text_for_ml(self.papers_df)
        
        #kind of a filter
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english', #the, an, etc.
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b' #minimum letters=2 and only letters allowed
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts) #fit (builds a dict with every work and their frequency) then transform (for each paper creat a vector with tf-idf score for each word)
        self.tfidf_similarity_matrix = cosine_similarity(self.tfidf_matrix) #use cosine angle between two vectors to calculate similarity between two papers
        
        print(f"✅ TF-IDF matrix: {self.tfidf_matrix.shape}")
        
        return self.tfidf_matrix
    
    def build_sentence_embeddings(self):
        """
        This captures MEANING, not just keywords:
        - "neural networks" ↔ "deep learning" (high similarity)
        - "machine learning" ↔ "artificial intelligence" (high similarity)
        - "computer vision" ↔ "image processing" (high similarity)
        
        """
        print("🧠 Building sentence transformer embeddings...")
        
        # Prepare text for semantic understanding
        semantic_texts = []
        for _, paper in self.papers_df.iterrows():
            # Focus on title + abstract for semantic meaning
            text_parts = []
            
            if paper.get('clean_title'):
                text_parts.append(paper['clean_title'])
            
            if paper.get('clean_abstract'):
                # Take first 200 words of abstract (transformer context limit)
                abstract_words = paper['clean_abstract'].split()[:200]
                text_parts.append(' '.join(abstract_words))
            
            semantic_text = '. '.join(text_parts)
            semantic_texts.append(semantic_text)
        
        # Generate embeddings 
        print(f"   🔄 Encoding {len(semantic_texts)} papers (this may take 1-2 minutes)...")
        self.sentence_embeddings = self.sentence_model.encode(
            semantic_texts, 
            show_progress_bar=True,
            batch_size=32  # Process in batches for efficiency
        )
        
        # Compute semantic similarity matrix
        self.semantic_similarity_matrix = cosine_similarity(self.sentence_embeddings) #ijth entry stands for similarity between paper i and j
        
        print(f"✅ Semantic embeddings: {self.sentence_embeddings.shape}")
        print(f"✅ Semantic similarity matrix: {self.semantic_similarity_matrix.shape}")
        
        # Show improvement over TF-IDF
        semantic_similarities = self.semantic_similarity_matrix[np.triu_indices_from(self.semantic_similarity_matrix, k=1)]
        print(f"   📈 Semantic similarity stats:")
        print(f"      Mean: {semantic_similarities.mean():.3f}")
        print(f"      Max:  {semantic_similarities.max():.3f}")
        print(f"      Min:  {semantic_similarities.min():.3f}")
        
        return self.sentence_embeddings
    
    def build_author_similarity(self):
        """
        Papers by same authors or research groups are often related!
        This captures collaboration networks and research continuity.
        """
        print("👥 Building author similarity matrix...")
        
        n_papers = len(self.papers_df)
        self.author_similarity_matrix = np.zeros((n_papers, n_papers))
        
        for i in range(n_papers):
            for j in range(i+1, n_papers):
                authors_i = set(self.papers_df.iloc[i]['authors'])
                authors_j = set(self.papers_df.iloc[j]['authors'])
                
                # Jaccard similarity for author overlap
                intersection = len(authors_i.intersection(authors_j))
                union = len(authors_i.union(authors_j))
                
                if union > 0:
                    author_sim = intersection / union
                    self.author_similarity_matrix[i, j] = author_sim
                    self.author_similarity_matrix[j, i] = author_sim
        
        author_similarities = self.author_similarity_matrix[np.triu_indices_from(self.author_similarity_matrix, k=1)]
        non_zero_author_sims = author_similarities[author_similarities > 0]
        
        print(f"✅ Author similarity matrix built")
        print(f"   📊 {len(non_zero_author_sims)} paper pairs share authors")
        if len(non_zero_author_sims) > 0:
            print(f"   📈 Author similarity stats (non-zero):")
            print(f"      Mean: {non_zero_author_sims.mean():.3f}")
            print(f"      Max:  {non_zero_author_sims.max():.3f}")
        
        return self.author_similarity_matrix
    
    def build_hybrid_similarity(self, tfidf_weight=0.3, semantic_weight=0.6, author_weight=0.1):
        """
        Combining:
        - TF-IDF (keyword matching): 30%
        - Semantic (meaning understanding): 60% 
        - Author networks: 10%
        
        """
        print(f"🔥 Building HYBRID similarity matrix...")
        print(f"   Weights: TF-IDF({tfidf_weight:.1f}) + Semantic({semantic_weight:.1f}) + Authors({author_weight:.1f})")
        
        # Ensure all components are built
        if self.tfidf_similarity_matrix is None:
            self.build_tfidf_vectors()
        
        if self.semantic_similarity_matrix is None:
            self.build_sentence_embeddings()
            
        if self.author_similarity_matrix is None:
            self.build_author_similarity()
        
        # Combine with weighted average
        self.hybrid_similarity_matrix = (
            tfidf_weight * self.tfidf_similarity_matrix +
            semantic_weight * self.semantic_similarity_matrix + 
            author_weight * self.author_similarity_matrix
        )
        
        # Show the improvement!
        hybrid_similarities = self.hybrid_similarity_matrix[np.triu_indices_from(self.hybrid_similarity_matrix, k=1)]
        print(f"✅ Hybrid similarity matrix: {self.hybrid_similarity_matrix.shape}")
        print(f"   📈 HYBRID similarity stats:")
        print(f"      Mean: {hybrid_similarities.mean():.3f}")
        print(f"      Max:  {hybrid_similarities.max():.3f}")
        print(f"      Min:  {hybrid_similarities.min():.3f}")
        
        # Compare with TF-IDF only
        tfidf_similarities = self.tfidf_similarity_matrix[np.triu_indices_from(self.tfidf_similarity_matrix, k=1)]
        print(f"   🆚 Improvement over TF-IDF:")
        print(f"      TF-IDF mean: {tfidf_similarities.mean():.3f} → Hybrid mean: {hybrid_similarities.mean():.3f}")
        
        return self.hybrid_similarity_matrix
    
    def find_similar_papers(self, paper_index, top_k=5, use_hybrid=True):
        """
        Enhanced recommendation function with hybrid similarity
        """
        # Choose which similarity matrix to use
        if use_hybrid and self.hybrid_similarity_matrix is not None:
            similarity_matrix = self.hybrid_similarity_matrix
            method = "HYBRID"
        else:
            if self.tfidf_similarity_matrix is None:
                self.build_tfidf_vectors()
            similarity_matrix = self.tfidf_similarity_matrix
            method = "TF-IDF"
        
        # Get similarity scores
        paper_similarities = similarity_matrix[paper_index]
        similar_indices = np.argsort(paper_similarities)[::-1][1:top_k+1]
        
        # Build recommendations with enhanced info
        recommendations = []
        for idx in similar_indices:
            paper = self.papers_df.iloc[idx]
            similarity_score = paper_similarities[idx]
            
            # Show breakdown of similarity sources if hybrid
            breakdown = {}
            if use_hybrid and self.hybrid_similarity_matrix is not None:
                breakdown = {
                    'tfidf': self.tfidf_similarity_matrix[paper_index, idx],
                    'semantic': self.semantic_similarity_matrix[paper_index, idx], 
                    'author': self.author_similarity_matrix[paper_index, idx]
                }
            
            recommendations.append({
                'paper_index': idx,
                'title': paper['title'],
                'similarity_score': similarity_score,
                'method': method,
                'breakdown': breakdown,
                'authors': paper['authors'][:3],
                'primary_category': paper['primary_category'],
                'keywords': paper['keywords'][:5],
                'url': paper['url']
            })
        
        return recommendations
    
    def recommend_by_title(self, search_title, top_k=5, use_hybrid=True):
        """Enhanced recommendation with hybrid similarity"""
        matches = self.papers_df[
            self.papers_df['title'].str.lower().str.contains(
                search_title.lower(), na=False
            )
        ]
        
        if matches.empty:
            print(f"❌ No papers found matching '{search_title}'")
            return []
        
        paper_index = matches.index[0]
        paper = matches.iloc[0]
        
        method = "HYBRID" if use_hybrid else "TF-IDF"
        print(f"🎯 Finding papers similar to (using {method}):")
        print(f"   📄 '{paper['title']}'")
        print(f"   👥 Authors: {', '.join(paper['authors'][:3])}")
        print(f"   🏷️  Category: {paper['primary_category']}")
        
        recommendations = self.find_similar_papers(paper_index, top_k, use_hybrid)
        return recommendations
    
    def display_enhanced_recommendations(self, recommendations):
        """Enhanced display with similarity breakdown"""
        print(f"\n🔥 TOP RECOMMENDATIONS ({recommendations[0]['method']}):")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. 📄 {rec['title']}")
            print(f"   🎯 Overall Similarity: {rec['similarity_score']:.3f}")
            
            # Show breakdown if available
            if rec['breakdown']:
                breakdown = rec['breakdown']
                print(f"   📊 Breakdown: TF-IDF({breakdown['tfidf']:.3f}) + Semantic({breakdown['semantic']:.3f}) + Authors({breakdown['author']:.3f})")
            
            print(f"   👥 Authors: {', '.join(rec['authors'])}")
            print(f"   🏷️  Category: {rec['primary_category']}")
            print(f"   🔑 Keywords: {', '.join(rec['keywords'])}")
    
    def compare_methods(self, search_title, top_k=3):
        """
        Compare TF-IDF vs Hybrid recommendations side by side!
        ====================================================
        
        This shows you the improvement from semantic understanding!
        """
        print(f"\n🆚 COMPARISON: TF-IDF vs HYBRID")
        print("=" * 80)
        
        # Get recommendations from both methods
        tfidf_recs = self.recommend_by_title(search_title, top_k, use_hybrid=False)
        print("\n" + "-" * 40)
        hybrid_recs = self.recommend_by_title(search_title, top_k, use_hybrid=True)
        
        print(f"\n📊 SIDE-BY-SIDE COMPARISON:")
        print("=" * 80)
        
        for i in range(min(len(tfidf_recs), len(hybrid_recs))):
            tfidf_rec = tfidf_recs[i]
            hybrid_rec = hybrid_recs[i]
            
            print(f"\n{i+1}. TF-IDF: {tfidf_rec['title'][:50]}... (sim: {tfidf_rec['similarity_score']:.3f})")
            print(f"   HYBRID: {hybrid_rec['title'][:50]}... (sim: {hybrid_rec['similarity_score']:.3f})")
            
            if tfidf_rec['paper_index'] != hybrid_rec['paper_index']:
                print("   ⚡ DIFFERENT recommendation!")

# Test the enhanced engine!
if __name__ == "__main__":
    print("🚀 Welcome to ENHANCED Paper Similarity Engine!")
    print("Now with Sentence Transformers + Author Networks!\n")
    
    # Initialize enhanced engine
    engine = EnhancedPaperSimilarityEngine()
    
    # Build all similarity components
    print("\n" + "="*80)
    print("🏗️  BUILDING ENHANCED SIMILARITY ENGINE")
    print("="*80)
    
    engine.build_tfidf_vectors()
    engine.build_sentence_embeddings()  # This is the new magic!
    engine.build_author_similarity()
    engine.build_hybrid_similarity()
    
    # Test enhanced recommendations
    print("\n" + "="*80)
    print("🎯 TESTING ENHANCED RECOMMENDATIONS")
    print("="*80)
    
    # Demo: Compare methods
    engine.compare_methods("neural", top_k=3)
    
    print("\n✨ Enhanced recommendation engine complete!")
    print("🎉 You now have semantic understanding in your recommender!")