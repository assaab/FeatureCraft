"""Text processing utilities for FeatureCraft with comprehensive NLP features."""

from __future__ import annotations

import re
import string
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline, FeatureUnion

from .logging import get_logger

logger = get_logger(__name__)


# ========== Basic Text Statistics ==========


class TextStatisticsExtractor(BaseEstimator, TransformerMixin):
    """Extract basic text statistics features.
    
    Features extracted:
    - char_count: Number of characters
    - word_count: Number of words
    - sentence_count: Number of sentences
    - avg_word_length: Average word length
    - unique_word_count: Number of unique words
    - unique_word_ratio: Ratio of unique words to total words
    """

    def __init__(self, extract_linguistic: bool = False, stopwords_language: str = "english"):
        """Initialize extractor.
        
        Args:
            extract_linguistic: Extract additional linguistic features
            stopwords_language: Language for stopwords detection
        """
        self.extract_linguistic = extract_linguistic
        self.stopwords_language = stopwords_language
        self.stopwords_set_: set[str] = set()

    def fit(self, X, y=None):
        """Fit extractor (load stopwords if needed)."""
        if self.extract_linguistic:
            try:
                from nltk.corpus import stopwords
                import nltk
                
                # Try to load stopwords, download if not available
                try:
                    self.stopwords_set_ = set(stopwords.words(self.stopwords_language))
                except LookupError:
                    logger.info(f"Downloading NLTK stopwords for {self.stopwords_language}...")
                    nltk.download('stopwords', quiet=True)
                    self.stopwords_set_ = set(stopwords.words(self.stopwords_language))
            except ImportError:
                logger.warning("NLTK not installed. Linguistic features will be limited.")
                self.stopwords_set_ = set()
        
        return self

    def transform(self, X):
        """Extract text statistics."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            # Take first column
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            # Assume array-like
            text_series = pd.Series(X).astype(str).fillna("")
        
        features = pd.DataFrame(index=text_series.index)
        
        # Basic statistics
        features['char_count'] = text_series.str.len()
        features['word_count'] = text_series.str.split().str.len()
        features['sentence_count'] = text_series.apply(self._count_sentences)
        features['avg_word_length'] = text_series.apply(self._avg_word_length)
        features['unique_word_count'] = text_series.apply(self._unique_word_count)
        features['unique_word_ratio'] = text_series.apply(self._unique_word_ratio)
        
        # Additional linguistic features
        if self.extract_linguistic:
            features['stopword_count'] = text_series.apply(
                lambda x: self._count_stopwords(x, self.stopwords_set_)
            )
            features['punctuation_count'] = text_series.apply(self._count_punctuation)
            features['uppercase_count'] = text_series.str.count(r'[A-Z]')
            features['uppercase_ratio'] = features['uppercase_count'] / features['char_count'].replace(0, 1)
            features['digit_count'] = text_series.str.count(r'\d')
            features['digit_ratio'] = features['digit_count'] / features['char_count'].replace(0, 1)
            features['special_char_count'] = text_series.apply(self._count_special_chars)
            features['whitespace_count'] = text_series.str.count(r'\s')
        
        # Fill NaN values with 0
        features = features.fillna(0)
        
        return features.values
    
    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences using simple heuristic."""
        if not text:
            return 0
        # Count periods, exclamation marks, question marks as sentence enders
        return len(re.findall(r'[.!?]+', text))
    
    @staticmethod
    def _avg_word_length(text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        return np.mean([len(w) for w in words])
    
    @staticmethod
    def _unique_word_count(text: str) -> int:
        """Count unique words."""
        words = text.lower().split()
        return len(set(words))
    
    @staticmethod
    def _unique_word_ratio(text: str) -> float:
        """Calculate ratio of unique words to total words."""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    @staticmethod
    def _count_stopwords(text: str, stopwords_set: set[str]) -> int:
        """Count stopwords in text."""
        if not stopwords_set:
            return 0
        words = text.lower().split()
        return sum(1 for w in words if w in stopwords_set)
    
    @staticmethod
    def _count_punctuation(text: str) -> int:
        """Count punctuation characters."""
        return sum(1 for c in text if c in string.punctuation)
    
    @staticmethod
    def _count_special_chars(text: str) -> int:
        """Count special characters (non-alphanumeric, non-whitespace, non-punctuation)."""
        return sum(1 for c in text if not (c.isalnum() or c.isspace() or c in string.punctuation))


# ========== Sentiment Analysis ==========


class SentimentAnalyzer(BaseEstimator, TransformerMixin):
    """Extract sentiment features from text.
    
    Features extracted:
    - sentiment_polarity: Polarity score (-1 to 1)
    - sentiment_subjectivity: Subjectivity score (0 to 1)
    """

    def __init__(self, method: str = "textblob"):
        """Initialize sentiment analyzer.
        
        Args:
            method: Sentiment analysis method ('textblob' or 'vader')
        """
        self.method = method
        self.analyzer_ = None

    def fit(self, X, y=None):
        """Fit analyzer (initialize sentiment tool)."""
        if self.method == "textblob":
            try:
                from textblob import TextBlob
                self.analyzer_ = "textblob"
            except ImportError:
                logger.warning("TextBlob not installed. Install with: pip install textblob")
                self.analyzer_ = None
        
        elif self.method == "vader":
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                import nltk
                
                # Download VADER lexicon if not available
                try:
                    self.analyzer_ = SentimentIntensityAnalyzer()
                except LookupError:
                    logger.info("Downloading VADER lexicon...")
                    nltk.download('vader_lexicon', quiet=True)
                    self.analyzer_ = SentimentIntensityAnalyzer()
            except ImportError:
                logger.warning("NLTK not installed for VADER. Install with: pip install nltk")
                self.analyzer_ = None
        
        return self

    def transform(self, X):
        """Extract sentiment features."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            text_series = pd.Series(X).astype(str).fillna("")
        
        if self.analyzer_ is None:
            # Return zeros if analyzer not available
            return np.zeros((len(text_series), 2))
        
        features = pd.DataFrame(index=text_series.index)
        
        if self.method == "textblob":
            sentiments = text_series.apply(self._analyze_textblob)
            features['sentiment_polarity'] = sentiments.apply(lambda x: x[0])
            features['sentiment_subjectivity'] = sentiments.apply(lambda x: x[1])
        
        elif self.method == "vader":
            sentiments = text_series.apply(self._analyze_vader)
            features['sentiment_polarity'] = sentiments.apply(lambda x: x[0])
            features['sentiment_compound'] = sentiments.apply(lambda x: x[1])
        
        return features.fillna(0).values
    
    def _analyze_textblob(self, text: str) -> tuple[float, float]:
        """Analyze sentiment using TextBlob."""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception:
            return 0.0, 0.0
    
    def _analyze_vader(self, text: str) -> tuple[float, float]:
        """Analyze sentiment using VADER."""
        try:
            scores = self.analyzer_.polarity_scores(text)
            # Return compound score and positive score
            return scores['compound'], scores['pos']
        except Exception:
            return 0.0, 0.0


# ========== Word Embeddings ==========


class WordEmbeddingFeatures(BaseEstimator, TransformerMixin):
    """Extract features using word embeddings (Word2Vec, GloVe, FastText).
    
    Aggregates word-level embeddings to document-level features.
    """

    def __init__(
        self,
        method: str = "word2vec",
        dims: int = 100,
        aggregation: str = "mean",
        pretrained_path: Optional[str] = None,
    ):
        """Initialize word embedding extractor.
        
        Args:
            method: Embedding method ('word2vec', 'glove', 'fasttext')
            dims: Embedding dimensionality
            aggregation: Aggregation method ('mean', 'max', 'sum')
            pretrained_path: Path to pretrained embeddings
        """
        self.method = method
        self.dims = dims
        self.aggregation = aggregation
        self.pretrained_path = pretrained_path
        self.embeddings_: dict[str, np.ndarray] = {}

    def fit(self, X, y=None):
        """Fit by loading pretrained embeddings."""
        if self.pretrained_path:
            logger.info(f"Loading pretrained {self.method} embeddings from {self.pretrained_path}...")
            self.embeddings_ = self._load_pretrained_embeddings(self.pretrained_path)
            logger.info(f"Loaded {len(self.embeddings_)} word embeddings.")
        else:
            logger.warning(
                f"No pretrained embeddings path provided. Word embeddings will not be used. "
                f"Download GloVe embeddings from https://nlp.stanford.edu/projects/glove/ "
                f"or train Word2Vec on your corpus."
            )
        
        return self

    def transform(self, X):
        """Transform text to embedding features."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            text_series = pd.Series(X).astype(str).fillna("")
        
        if not self.embeddings_:
            # Return zeros if no embeddings loaded
            return np.zeros((len(text_series), self.dims))
        
        # Extract embedding features for each document
        embeddings = text_series.apply(self._text_to_embedding)
        return np.vstack(embeddings.values)
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        words = text.lower().split()
        word_vectors = [self.embeddings_[w] for w in words if w in self.embeddings_]
        
        if not word_vectors:
            # Return zero vector if no words found
            return np.zeros(self.dims)
        
        word_vectors = np.array(word_vectors)
        
        # Aggregate word vectors
        if self.aggregation == "mean":
            return np.mean(word_vectors, axis=0)
        elif self.aggregation == "max":
            return np.max(word_vectors, axis=0)
        elif self.aggregation == "sum":
            return np.sum(word_vectors, axis=0)
        else:
            return np.mean(word_vectors, axis=0)
    
    def _load_pretrained_embeddings(self, path: str) -> dict[str, np.ndarray]:
        """Load pretrained embeddings from file (GloVe format)."""
        embeddings = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    if len(vector) == self.dims:
                        embeddings[word] = vector
        except Exception as e:
            logger.error(f"Failed to load embeddings from {path}: {e}")
        
        return embeddings


# ========== Sentence Embeddings (Transformers) ==========


class SentenceEmbeddingFeatures(BaseEstimator, TransformerMixin):
    """Extract sentence embeddings using SentenceTransformers.
    
    Uses pre-trained transformer models for contextual embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_length: int = 128,
    ):
        """Initialize sentence embedding extractor.
        
        Args:
            model_name: SentenceTransformer model name
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_ = None

    def fit(self, X, y=None):
        """Fit by loading the model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model_ = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model_.get_sentence_embedding_dimension()}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
            self.model_ = None
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            self.model_ = None
        
        return self

    def transform(self, X):
        """Transform text to sentence embeddings."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            text_series = pd.Series(X).astype(str).fillna("")
        
        if self.model_ is None:
            # Return zeros if model not available
            return np.zeros((len(text_series), 384))  # Default dimension
        
        # Encode texts in batches
        texts = text_series.tolist()
        embeddings = self.model_.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        
        return embeddings


# ========== Named Entity Recognition ==========


class NERFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract named entity features using spaCy.
    
    Features extracted:
    - entity_count: Total number of entities
    - person_count, org_count, gpe_count, etc.: Counts per entity type
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        entity_types: Optional[list[str]] = None,
    ):
        """Initialize NER extractor.
        
        Args:
            model_name: spaCy model name
            entity_types: Entity types to extract (default: common types)
        """
        self.model_name = model_name
        self.entity_types = entity_types or ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY"]
        self.nlp_ = None

    def fit(self, X, y=None):
        """Fit by loading spaCy model."""
        try:
            import spacy
            
            logger.info(f"Loading spaCy model: {self.model_name}")
            try:
                self.nlp_ = spacy.load(self.model_name)
            except OSError:
                logger.warning(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Download with: python -m spacy download {self.model_name}"
                )
                self.nlp_ = None
        except ImportError:
            logger.warning("spaCy not installed. Install with: pip install spacy")
            self.nlp_ = None
        
        return self

    def transform(self, X):
        """Extract NER features."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            text_series = pd.Series(X).astype(str).fillna("")
        
        if self.nlp_ is None:
            # Return zeros if model not available
            return np.zeros((len(text_series), len(self.entity_types) + 1))
        
        features = pd.DataFrame(index=text_series.index)
        
        # Extract entity counts
        entity_counts = text_series.apply(self._extract_entities)
        
        features['entity_count'] = entity_counts.apply(lambda x: sum(x.values()))
        for ent_type in self.entity_types:
            features[f'{ent_type.lower()}_count'] = entity_counts.apply(lambda x: x.get(ent_type, 0))
        
        return features.fillna(0).values
    
    def _extract_entities(self, text: str) -> dict[str, int]:
        """Extract entity counts from text."""
        try:
            doc = self.nlp_(text[:1000000])  # Limit text length for performance
            entity_counts = {}
            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
            return entity_counts
        except Exception:
            return {}


# ========== Topic Modeling ==========


class TopicModelingFeatures(BaseEstimator, TransformerMixin):
    """Extract topic distribution features using LDA.
    
    Fits LDA on the corpus and transforms texts to topic probabilities.
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_features: int = 5000,
        random_state: int = 42,
    ):
        """Initialize topic modeling extractor.
        
        Args:
            n_topics: Number of topics
            max_features: Maximum vocabulary size
            random_state: Random seed
        """
        self.n_topics = n_topics
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer_ = None
        self.lda_ = None

    def fit(self, X, y=None):
        """Fit LDA model on corpus."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            text_series = pd.Series(X).astype(str).fillna("")
        
        texts = text_series.tolist()
        
        # Vectorize text
        self.vectorizer_ = CountVectorizer(
            max_features=self.max_features,
            stop_words='english',
            min_df=2,
        )
        
        try:
            dtm = self.vectorizer_.fit_transform(texts)
            
            # Fit LDA
            self.lda_ = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=10,
                n_jobs=-1,
            )
            self.lda_.fit(dtm)
            
            logger.info(f"Fitted LDA with {self.n_topics} topics on {len(texts)} documents.")
        except Exception as e:
            logger.warning(f"LDA fitting failed: {e}")
            self.lda_ = None
        
        return self

    def transform(self, X):
        """Transform text to topic distributions."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            text_series = pd.Series(X).astype(str).fillna("")
        
        if self.lda_ is None or self.vectorizer_ is None:
            # Return zeros if model not fitted
            return np.zeros((len(text_series), self.n_topics))
        
        texts = text_series.tolist()
        
        try:
            dtm = self.vectorizer_.transform(texts)
            topic_dist = self.lda_.transform(dtm)
            return topic_dist
        except Exception as e:
            logger.warning(f"LDA transform failed: {e}")
            return np.zeros((len(text_series), self.n_topics))


# ========== Readability Scores ==========


class ReadabilityScoreExtractor(BaseEstimator, TransformerMixin):
    """Extract readability scores (Flesch-Kincaid, SMOG, etc.).
    
    Features extracted depend on the metrics parameter.
    """

    def __init__(self, metrics: Optional[list[str]] = None):
        """Initialize readability extractor.
        
        Args:
            metrics: List of readability metrics to compute
        """
        self.metrics = metrics or ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index"]

    def fit(self, X, y=None):
        """Fit (no-op)."""
        return self

    def transform(self, X):
        """Extract readability scores."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            text_series = pd.Series(X).astype(str).fillna("")
        
        features = pd.DataFrame(index=text_series.index)
        
        # Try to use textstat library
        try:
            import textstat
            
            if "flesch_reading_ease" in self.metrics:
                features['flesch_reading_ease'] = text_series.apply(
                    lambda x: textstat.flesch_reading_ease(x) if x else 0
                )
            
            if "flesch_kincaid_grade" in self.metrics:
                features['flesch_kincaid_grade'] = text_series.apply(
                    lambda x: textstat.flesch_kincaid_grade(x) if x else 0
                )
            
            if "smog_index" in self.metrics:
                features['smog_index'] = text_series.apply(
                    lambda x: textstat.smog_index(x) if x else 0
                )
            
            if "coleman_liau_index" in self.metrics:
                features['coleman_liau_index'] = text_series.apply(
                    lambda x: textstat.coleman_liau_index(x) if x else 0
                )
            
            if "automated_readability_index" in self.metrics:
                features['automated_readability_index'] = text_series.apply(
                    lambda x: textstat.automated_readability_index(x) if x else 0
                )
        
        except ImportError:
            logger.warning(
                "textstat not installed. Readability scores will be zeros. "
                "Install with: pip install textstat"
            )
            for metric in self.metrics:
                features[metric] = 0
        except Exception as e:
            logger.warning(f"Readability score extraction failed: {e}")
            for metric in self.metrics:
                features[metric] = 0
        
        return features.fillna(0).values


# ========== Text Preprocessing ==========


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess text with various cleaning operations."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        stem: bool = False,
        stopwords_language: str = "english",
    ):
        """Initialize text preprocessor.
        
        Args:
            lowercase: Convert to lowercase
            remove_special_chars: Remove special characters
            remove_stopwords: Remove stopwords
            lemmatize: Apply lemmatization
            stem: Apply stemming
            stopwords_language: Language for stopwords
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.stopwords_language = stopwords_language
        self.stopwords_set_: set[str] = set()
        self.stemmer_ = None
        self.nlp_ = None

    def fit(self, X, y=None):
        """Fit preprocessor (load tools if needed)."""
        if self.remove_stopwords:
            try:
                from nltk.corpus import stopwords
                import nltk
                
                try:
                    self.stopwords_set_ = set(stopwords.words(self.stopwords_language))
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                    self.stopwords_set_ = set(stopwords.words(self.stopwords_language))
            except ImportError:
                logger.warning("NLTK not installed for stopwords removal")
        
        if self.stem:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer_ = PorterStemmer()
            except ImportError:
                logger.warning("NLTK not installed for stemming")
        
        if self.lemmatize:
            try:
                import spacy
                self.nlp_ = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except Exception:
                logger.warning("spaCy not available for lemmatization")
        
        return self

    def transform(self, X):
        """Preprocess text."""
        # Handle Series or single column
        if isinstance(X, pd.Series):
            text_series = X.astype(str).fillna("")
        elif isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0].astype(str).fillna("")
        else:
            text_series = pd.Series(X).astype(str).fillna("")
        
        # Apply preprocessing
        processed = text_series.copy()
        
        if self.lowercase:
            processed = processed.str.lower()
        
        if self.remove_special_chars:
            processed = processed.str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
        
        if self.remove_stopwords and self.stopwords_set_:
            processed = processed.apply(lambda x: self._remove_stopwords(x, self.stopwords_set_))
        
        if self.stem and self.stemmer_:
            processed = processed.apply(self._apply_stemming)
        
        if self.lemmatize and self.nlp_:
            processed = processed.apply(self._apply_lemmatization)
        
        # Return as Series for downstream processing
        return processed
    
    @staticmethod
    def _remove_stopwords(text: str, stopwords_set: set[str]) -> str:
        """Remove stopwords from text."""
        words = text.split()
        return ' '.join([w for w in words if w not in stopwords_set])
    
    def _apply_stemming(self, text: str) -> str:
        """Apply stemming to text."""
        words = text.split()
        return ' '.join([self.stemmer_.stem(w) for w in words])
    
    def _apply_lemmatization(self, text: str) -> str:
        """Apply lemmatization to text."""
        doc = self.nlp_(text)
        return ' '.join([token.lemma_ for token in doc])


# ========== Adaptive SVD (from original) ==========


class AdaptiveSVD(BaseEstimator, TransformerMixin):
    """SVD that adapts n_components to input feature count."""

    def __init__(self, n_components: int = 100, random_state: int | None = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd_: TruncatedSVD | None = None
        self.n_components_: int = 0

    def fit(self, X, y=None):
        """Fit SVD with adaptive component count."""
        n_features = X.shape[1]
        # SVD requires n_components < min(n_samples, n_features)
        self.n_components_ = min(self.n_components, n_features - 1, X.shape[0] - 1)
        if self.n_components_ < 1:
            # Skip SVD if we can't reduce dimensionality
            self.svd_ = None
        else:
            self.svd_ = TruncatedSVD(
                n_components=self.n_components_, random_state=self.random_state
            )
            self.svd_.fit(X)
        return self

    def transform(self, X):
        """Transform using fitted SVD or pass-through."""
        if self.svd_ is None:
            # Return as-is if SVD wasn't applicable
            return X
        return self.svd_.transform(X)


# ========== Pipeline Builder ==========


def build_text_pipeline(
    column_name: str,
    max_features: int = 20000,
    svd_components: int | None = None,
    use_hashing: bool = False,
    hashing_features: int = 16384,
    char_ngrams: bool = False,
    ngram_range: tuple[int, int] = (1, 2),
    remove_stopwords: bool = False,
    min_df: int = 1,
) -> Pipeline:
    """Build text processing pipeline with TF-IDF/Hashing.

    Args:
        column_name: Name of the text column (for reference)
        max_features: Maximum TF-IDF features
        svd_components: Target SVD components (adaptively adjusted to data)
        use_hashing: Use HashingVectorizer instead of TF-IDF
        hashing_features: Number of features for hashing
        char_ngrams: Use character n-grams instead of word n-grams
        ngram_range: N-gram range for text vectorization
        remove_stopwords: Remove stopwords in vectorization
        min_df: Minimum document frequency
    """
    steps = []
    
    stopwords = 'english' if remove_stopwords else None
    
    if use_hashing:
        # Memory-efficient hashing vectorizer
        if char_ngrams:
            vectorizer = HashingVectorizer(
                n_features=hashing_features,
                analyzer="char_wb",
                ngram_range=(3, 5),
                alternate_sign=False,
            )
        else:
            vectorizer = HashingVectorizer(
                n_features=hashing_features,
                ngram_range=ngram_range,
                alternate_sign=False,
            )
        steps.append(("hasher", vectorizer))
        logger.debug(f"Using HashingVectorizer with {hashing_features} features for '{column_name}'")
    else:
        # Standard TF-IDF
        if char_ngrams:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=max_features,
                stop_words=stopwords,
                min_df=min_df,
            )
        else:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                stop_words=stopwords,
                min_df=min_df,
            )
        steps.append(("tfidf", vectorizer))
        logger.debug(f"Using TF-IDF with max_features={max_features} for '{column_name}'")
    
    if svd_components:
        # Use AdaptiveSVD to handle cases where vectorizer produces fewer features
        steps.append(("svd", AdaptiveSVD(n_components=svd_components, random_state=42)))
    
    return Pipeline(steps=steps)


def build_comprehensive_text_pipeline(
    column_name: str,
    cfg: Any,
) -> Pipeline:
    """Build comprehensive text pipeline with all NLP features.
    
    Args:
        column_name: Name of the text column
        cfg: FeatureCraftConfig instance
    
    Returns:
        Pipeline with all configured text transformers
    """
    transformers = []
    
    # 1. Text Statistics
    if cfg.text_extract_statistics:
        transformers.append((
            'statistics',
            TextStatisticsExtractor(
                extract_linguistic=cfg.text_extract_linguistic,
                stopwords_language=cfg.text_stopwords_language,
            )
        ))
    
    # 2. Sentiment Analysis
    if cfg.text_extract_sentiment:
        transformers.append((
            'sentiment',
            SentimentAnalyzer(method=cfg.text_sentiment_method)
        ))
    
    # 3. Word Embeddings
    if cfg.text_use_word_embeddings:
        transformers.append((
            'word_embeddings',
            WordEmbeddingFeatures(
                method=cfg.text_embedding_method,
                dims=cfg.text_embedding_dims,
                aggregation=cfg.text_embedding_aggregation,
                pretrained_path=cfg.text_embedding_pretrained_path,
            )
        ))
    
    # 4. Sentence Embeddings
    if cfg.text_use_sentence_embeddings:
        transformers.append((
            'sentence_embeddings',
            SentenceEmbeddingFeatures(
                model_name=cfg.text_sentence_model,
                batch_size=cfg.text_sentence_batch_size,
                max_length=cfg.text_sentence_max_length,
            )
        ))
    
    # 5. NER Features
    if cfg.text_extract_ner:
        transformers.append((
            'ner',
            NERFeatureExtractor(
                model_name=cfg.text_ner_model,
                entity_types=cfg.text_ner_entity_types,
            )
        ))
    
    # 6. Topic Modeling
    if cfg.text_use_topic_modeling:
        transformers.append((
            'topics',
            TopicModelingFeatures(
                n_topics=cfg.text_topic_n_topics,
                max_features=cfg.text_topic_max_features,
                random_state=cfg.random_state,
            )
        ))
    
    # 7. Readability Scores
    if cfg.text_extract_readability:
        transformers.append((
            'readability',
            ReadabilityScoreExtractor(metrics=cfg.text_readability_metrics)
        ))
    
    # 8. TF-IDF/Bag-of-Words (always include for basic text representation)
    transformers.append((
        'tfidf',
        build_text_pipeline(
            column_name=column_name,
            max_features=cfg.tfidf_max_features,
            svd_components=cfg.svd_components_for_trees if cfg.svd_components_for_trees > 0 else None,
            use_hashing=cfg.text_use_hashing,
            hashing_features=cfg.text_hashing_features,
            char_ngrams=cfg.text_char_ngrams,
            ngram_range=cfg.ngram_range,
            remove_stopwords=cfg.text_remove_stopwords,
            min_df=cfg.text_min_word_freq,
        )
    ))
    
    # Combine all transformers using FeatureUnion
    if len(transformers) > 1:
        feature_union = FeatureUnion(transformers, n_jobs=1)
        return Pipeline([('features', feature_union)])
    else:
        # Only TF-IDF, return directly
        return transformers[0][1]
