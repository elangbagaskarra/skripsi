# sentiment_dict.py
POSITIVE_WORDS = {
    "baik", "bagus", "senang", "indah", "hebat", "suka", "luar biasa", "mantap", "keren", "puas"
}

NEGATIVE_WORDS = {
    "buruk", "jelek", "sedih", "marah", "kecewa", "parah", "gagal", "mengecewakan", "sakit", "payah"
}

def determine_sentiment(text, stemmer):
    # Stem teks ke bentuk dasar menggunakan Sastrawi
    stemmed_text = stemmer.stem(text.lower())
    words = stemmed_text.split()

    # Hitung skor sentimen
    positive_score = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_score = sum(1 for word in words if word in NEGATIVE_WORDS)

    # Tentukan sentimen berdasarkan skor
    if positive_score > negative_score:
        return "positif"
    elif negative_score > positive_score:
        return "negatif"
    else:
        return "netral"