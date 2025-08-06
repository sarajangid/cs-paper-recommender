"""
CS Paper Recommender - arXiv Data Collection & Feature Engineering
==================================================================

This script demonstrates core ML concepts:
1. Feature Engineering: Converting raw text into ML-ready features
2. Data Pipeline: Systematic data collection and preprocessing
3. API Integration: Real-world data sourcing

"""

import arxiv
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import re
from collections import Counter
import matplotlib.pyplot as plt

class CSPaperScraper:
    def __init__(self):
        """
        Initialize our paper scraper.
        
        """
        # CS categories from arXiv - this defines our feature space
        self.cs_categories = {
            'cs.AI': 'Artificial Intelligence',
            'cs.LG': 'Machine Learning', 
            'cs.CV': 'Computer Vision',
            'cs.CL': 'Natural Language Processing',
            'cs.CR': 'Cryptography and Security',
            'cs.DB': 'Databases',
            'cs.DC': 'Distributed Computing',
            'cs.DS': 'Data Structures and Algorithms',
            'cs.HC': 'Human-Computer Interaction',
            'cs.IR': 'Information Retrieval',
            'cs.NI': 'Networking',
            'cs.OS': 'Operating Systems',
            'cs.PL': 'Programming Languages',
            'cs.SE': 'Software Engineering',
            'cs.SY': 'Systems and Control'
        }
        
        self.papers = []
        print("🚀 CS Paper Scraper initialized!")
        print(f"📊 Targeting {len(self.cs_categories)} CS categories")
    
    def extract_features_from_paper(self, paper):
        """
        CORE ML CONCEPT: Feature Engineering
        ====================================
        
        Raw paper data is messy. We need to extract structured features
        that our ML algorithms can understand and learn from.
        
        Think of this as teaching the computer what makes papers similar!
        """
        
        # Feature 1: Text Features (for content-based recommendations)
        title = paper.title.strip()
        abstract = paper.summary.strip()
        
        # Feature 2: Author Features (for collaborative filtering)
        authors = [author.name for author in paper.authors]
        
        # Feature 3: Category Features (multi-hot encoding)
        categories = [cat.strip() for cat in paper.categories]
        primary_category = paper.primary_category
        
        # Feature 4: Temporal Features (papers have trends!)
        pub_date = paper.published.date()
        
        # Feature 5: Engagement Features (proxy for quality)
        # Note: arXiv doesn't give citations directly, but we can use other signals
        
        # Feature Engineering: Clean and structure the text
        clean_title = self.clean_text(title)
        clean_abstract = self.clean_text(abstract)
        
        # Feature Engineering: Extract keywords from title and abstract
        keywords = self.extract_keywords(clean_title + ' ' + clean_abstract)
        
        return {
            'id': paper.entry_id.split('/')[-1],  # Unique identifier
            'title': title,
            'abstract': abstract,
            'clean_title': clean_title,
            'clean_abstract': clean_abstract,
            'authors': authors,
            'author_count': len(authors),
            'categories': categories,
            'primary_category': primary_category,
            'keywords': keywords,
            'published_date': pub_date.isoformat(),
            'year': pub_date.year,
            'month': pub_date.month,
            'url': paper.entry_id,
            'pdf_url': paper.pdf_url
        }
    
    def clean_text(self, text):
        """
        Text Preprocessing for NLP/ML
        =============================
        
        Raw text is noisy! We need to clean it for our ML models.
        This is standard preprocessing in any text-based ML system.
        """
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?]', '', text)
        
        # Convert to lowercase for consistency
        text = text.lower().strip()
        
        return text
    
    def extract_keywords(self, text, top_n=10):
        """
        Keyword Extraction - Simple but Effective Feature Engineering
        ============================================================
        
        We're extracting the most important words as features.
        Later, we'll use more sophisticated methods like TF-IDF.
        """
        # Remove common stop words (basic version)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                     'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'can',
                     'this', 'that', 'these', 'those', 'we', 'our', 'paper',
                     'approach', 'method', 'using', 'used', 'show', 'present'}
        
        # Split into words and count frequencies
        words = text.split()
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Get most common words as keywords
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(top_n)]
        
        return keywords
    
    def scrape_papers(self, max_papers=1000, days_back=365):
        """
        Data Collection Pipeline
        ========================
        
        This is where we gather our training data!
        In ML, data quality > algorithm complexity.
        """
        print(f"🔍 Starting to scrape {max_papers} papers from last {days_back} days...")
        
        # Calculate date range (recent papers are more relevant)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        papers_per_category = max_papers // len(self.cs_categories)
        
        for category, description in self.cs_categories.items():
            print(f"\n📚 Scraping {category} ({description})...")
            
            try:
                # Build search query
                search_query = f"cat:{category}"
                
                # Create search client
                search = arxiv.Search(
                    query=search_query,
                    max_results=papers_per_category,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                # Collect papers
                category_papers = []
                for paper in search.results():
                    # Filter by date
                    if paper.published.date() >= start_date.date():
                        features = self.extract_features_from_paper(paper)
                        category_papers.append(features)
                    
                    # Rate limiting - be nice to arXiv!
                    time.sleep(0.1)
                
                self.papers.extend(category_papers)
                print(f"✅ Collected {len(category_papers)} papers from {category}")
                
            except Exception as e:
                print(f"❌ Error scraping {category}: {str(e)}")
                continue
        
        print(f"\n🎉 Total papers collected: {len(self.papers)}")
        return self.papers
    
    def save_data(self, filename='data/raw/cs_papers.json'):
        """
        Data Persistence - Save our precious training data!
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.papers, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(self.papers)} papers to {filename}")
    
    def create_dataframe(self):
        """
        Convert to pandas DataFrame - the standard format for ML in Python
        
        DataFrames make feature engineering and analysis much easier!
        """
        df = pd.DataFrame(self.papers)
        print(f"📊 Created DataFrame with shape: {df.shape}")
        return df
    
    def analyze_data(self, df):
        """
        Exploratory Data Analysis (EDA) - Understanding our features
        ============================================================
        
        Before building ML models, we MUST understand our data!
        This is where we discover patterns and validate our features.
        """
        print("\n🔍 DATA ANALYSIS REPORT")
        print("=" * 50)
        
        # Basic statistics
        print(f"📊 Total papers: {len(df)}")
        print(f"📅 Date range: {df['published_date'].min()} to {df['published_date'].max()}")
        print(f"👥 Unique authors: {df['authors'].apply(len).sum()}")
        
        # Category distribution
        print(f"\n📚 Papers by category:")
        category_counts = df['primary_category'].value_counts()
        for cat, count in category_counts.head(10).items():
            category_name = self.cs_categories.get(cat, cat)
            print(f"  {cat} ({category_name}): {count}")
        
        # Author statistics
        print(f"\n👥 Author statistics:")
        print(f"  Average authors per paper: {df['author_count'].mean():.1f}")
        print(f"  Max authors on a paper: {df['author_count'].max()}")
        
        # Temporal patterns
        print(f"\n📅 Temporal patterns:")
        yearly_counts = df['year'].value_counts().sort_index()
        for year, count in yearly_counts.items():
            print(f"  {year}: {count} papers")

# Let's test our scraper!
if __name__ == "__main__":
    print("🚀 Welcome to CS Paper Recommender!")
    print("This is your first step into ML feature engineering!\n")
    
    # Initialize scraper
    scraper = CSPaperScraper()
    
    # Start small for testing (increase later)
    papers = scraper.scrape_papers(max_papers=100, days_back=30)
    
    # Save the data
    scraper.save_data()
    
    # Convert to DataFrame for analysis
    df = scraper.create_dataframe()
    
    # Analyze our features
    scraper.analyze_data(df)
    
    print("\n✨ Feature engineering complete!")
    print("Next: We'll build recommendation models using these features!")