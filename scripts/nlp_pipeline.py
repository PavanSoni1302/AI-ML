import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# 1. SETUP
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

# 2. TEXT CLEANING FUNCTION
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    
    return " ".join(filtered)

# 3. SAMPLE BIOS
bios = [
    "I love hiking and outdoor adventures!",
    "Passionate about machine learning and AI.",
    "Food lover and professional chef.",
    "I enjoy coding late at night and solving problems.",
    "Fitness enthusiast and sports lover."
]

# 4. CLEAN BIOS
cleaned_bios = [clean_text(bio) for bio in bios]

print("\n🧹 Cleaned Bios:")
for bio in cleaned_bios:
    print(bio)

# 5. TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(cleaned_bios)

print("\n Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\n Matrix Shape:", tfidf_matrix.shape)

# 6. INTEREST MAPPING (TOP WORDS)
import numpy as np

feature_names = vectorizer.get_feature_names_out()

for i, row in enumerate(tfidf_matrix.toarray()):
    top_indices = row.argsort()[-3:][::-1]
    top_words = [feature_names[j] for j in top_indices]
    
    print(f"\nUser {i+1} Top Interests:", top_words)

# 7. SENTIMENT ANALYSIS
print("\n Sentiment Analysis:")

for bio in bios:
    analysis = TextBlob(bio)
    
    print(f"\nText: {bio}")
    print("Polarity:", round(analysis.sentiment.polarity, 2))
    print("Subjectivity:", round(analysis.sentiment.subjectivity, 2))