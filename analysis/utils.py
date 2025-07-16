import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Case Folding: Mengubah semua huruf jadi kecil
def case_folding(text):
    return text.lower() if isinstance(text, str) else text

# Cleansing: Menghapus karakter selain huruf dan spasi (tanda baca, angka, simbol)
def cleansing(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hanya huruf dan spasi yang disisakan
    text = re.sub(r'\s+', ' ', text)  # Menormalkan spasi ganda jadi satu
    return text.strip()

# Tokenizing: Memecah kalimat jadi list kata
def tokenizing(text):
    return text.split()

def load_stopwords_from_file(filepath):
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stopwords.add(word)
    return stopwords



# Stemming: Mengubah kata jadi bentuk dasar (root word)
def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(token) for token in tokens]
def stopword_removal(tokens, stopword_list):
    return [token for token in tokens if token not in stopword_list]

