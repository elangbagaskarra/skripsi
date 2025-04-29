import os
import string
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from .models import Dataset
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tweepy
import instaloader
from imblearn.over_sampling import RandomOverSampler

# Setup logging
logger = logging.getLogger(__name__)

# Inisialisasi stemmer dan stopword Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

# Kredensial Twitter API (ganti dengan kredensial Anda)
BEARER_TOKEN = "YOUR_BEARER_TOKEN"  # Ganti dengan token Anda
INSTALOADER = instaloader.Instaloader()

# Load lexicon dan kamus slang/stop words
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEXICON_PATH = os.path.join(BASE_DIR, 'lexicon', 'modified_full_lexicon.csv')
SLANG_PATH = os.path.join(BASE_DIR, 'lexicon', 'slang.csv')
STOPWORD_PATH = os.path.join(BASE_DIR, 'lexicon', 'stopword.txt')

# Load lexicon
try:
    lexicon_df = pd.read_csv(LEXICON_PATH)
    lexicon_dict = dict(zip(lexicon_df['word'], lexicon_df['weight']))
except Exception as e:
    logger.error(f"Gagal memuat lexicon: {str(e)}")
    lexicon_dict = {}

# Load slang dictionary
slang_dict = {}
try:
    with open(SLANG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if ',' in line:
                slang, formal = line.strip().split(',', 1)
                slang_dict[slang] = formal
except Exception as e:
    logger.error(f"Gagal memuat slang dictionary: {str(e)}")

# Load stop words
custom_stopwords = set()
try:
    with open(STOPWORD_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            custom_stopwords.add(line.strip())
except Exception as e:
    logger.error(f"Gagal memuat stop words: {str(e)}")

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Fungsi normalisasi slang
def normalize_slang(text):
    words = text.split()
    normalized = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized)

# Fungsi preprocessing teks
def preprocess_text(text):
    if not isinstance(text, str) or not text or text.strip() == "":
        logger.debug(f"Teks tidak valid: {text}")
        return []
    text = text.lower()
    text = normalize_slang(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = stopword.remove(text)
    text = stemmer.stem(text)
    tokens = word_tokenize(text)
    # Izinkan token alfanumerik untuk menjaga kata-kata kunci
    tokens = [t for t in tokens if t not in custom_stopwords]
    logger.debug(f"Teks: {text}, Tokens: {tokens}")
    return tokens

# Fungsi penentuan sentimen menggunakan lexicon
def determine_sentiment(text):
    tokens = preprocess_text(text)
    sentiment_score = sum(lexicon_dict.get(token, 0) for token in tokens)
    logger.debug(f"Teks: {text}, Skor sentimen: {sentiment_score}")
    if sentiment_score > 0:
        return 'positif'
    elif sentiment_score < 0:
        return 'negatif'
    return 'netral'

# Fungsi pelatihan dan evaluasi model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="both"):
    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),  # Alpha kecil untuk dataset kecil
        "SVM": LinearSVC(max_iter=1000)
    }
    metrics = {}
    predictions = {}
    for name, model in models.items():
        if model_type != "both" and name != model_type:
            continue
        try:
            # Oversampling untuk menangani data tidak seimbang
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            metrics[name] = {
                'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
                'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
                'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
                'f1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            }
            logger.info(f"Metrik {name}: {metrics[name]}")
        except Exception as e:
            logger.error(f"Gagal melatih model {name}: {str(e)}")
            metrics[name] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    return metrics, predictions

# Views untuk Dataset
def dataset_view(request):
    search_text = request.GET.get('search_text', '')
    search_sentiment = request.GET.get('search_sentiment', '')

    dataset = Dataset.objects.all()
    if search_text:
        dataset = dataset.filter(text__icontains=search_text)
    if search_sentiment:
        dataset = dataset.filter(sentiment=search_sentiment)

    return render(request, 'analysis/dataset.html', {
        'dataset': dataset,
        'search_text': search_text,
        'search_sentiment': search_sentiment
    })

def add_data(request):
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        sentiment = request.POST.get('sentiment', '').strip()
        if text and sentiment in ['positif', 'negatif', 'netral']:
            stemmed_text = stemmer.stem(text)
            if stemmed_text:
                Dataset.objects.create(text=text, sentiment=sentiment)  # Simpan teks asli
                messages.success(request, 'Data berhasil ditambahkan.')
            else:
                messages.error(request, 'Teks tidak valid setelah stemming.')
        else:
            messages.error(request, 'Data tidak valid.')
    return redirect('analysis:dataset')

def edit_data(request, data_id):
    data = get_object_or_404(Dataset, id=data_id)
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        sentiment = request.POST.get('sentiment', '').strip()
        if text and sentiment in ['positif', 'negatif', 'netral']:
            data.text = text  # Simpan teks asli
            data.sentiment = sentiment
            data.save()
            messages.success(request, 'Data berhasil diperbarui.')
        else:
            messages.error(request, 'Data tidak valid.')
    return redirect('analysis:dataset')

def delete_data(request, data_id):
    data = get_object_or_404(Dataset, id=data_id)
    data.delete()
    messages.success(request, 'Data berhasil dihapus.')
    return redirect('analysis:dataset')

# Views untuk Crawling
def crawl_twitter(request):
    if request.method == 'POST':
        keyword = request.POST.get('twitter_keyword', '').strip()
        tweet_limit = int(request.POST.get('tweet_limit', 10))
        tweet_limit = min(tweet_limit, 50)

        try:
            client = tweepy.Client(bearer_token=BEARER_TOKEN)
            tweets = client.search_recent_tweets(query=keyword, max_results=tweet_limit, tweet_fields=["created_at"])
            count = 0
            for tweet in tweets.data:
                text = tweet.text
                sentiment = determine_sentiment(text)
                if not Dataset.objects.filter(text=text, sentiment=sentiment).exists():
                    Dataset.objects.create(text=text, sentiment=sentiment)
                    count += 1
            messages.success(request, f'Berhasil mengambil {count} tweets dari Twitter.')
        except Exception as e:
            logger.error(f"Gagal mengambil data Twitter: {str(e)}")
            messages.error(request, f'Gagal mengambil data Twitter: {str(e)}')
    return redirect('analysis:dataset')

def crawl_instagram(request):
    if request.method == 'POST':
        hashtag = request.POST.get('instagram_hashtag', '').strip()
        post_limit = int(request.POST.get('post_limit', 10))
        post_limit = min(post_limit, 50)

        try:
            posts = instaloader.Hashtag.from_name(INSTALOADER.context, hashtag).get_posts()
            count = 0
            for post in posts:
                if count >= post_limit:
                    break
                if post.caption:
                    text = post.caption
                    sentiment = determine_sentiment(text)
                    if not Dataset.objects.filter(text=text, sentiment=sentiment).exists():
                        Dataset.objects.create(text=text, sentiment=sentiment)
                        count += 1
            messages.success(request, f'Berhasil mengambil {count} postingan dari Instagram.')
        except Exception as e:
            logger.error(f"Gagal mengambil data Instagram: {str(e)}")
            messages.error(request, f'Gagal mengambil data Instagram: {str(e)}')
    return redirect('analysis:dataset')

def upload_excel(request):
    if request.method == 'POST' and 'excel_file' in request.FILES:
        excel_file = request.FILES['excel_file']
        try:
            df = pd.read_excel(excel_file)
            if 'text' not in df.columns:
                messages.error(request, "File Excel harus memiliki kolom 'text'.")
                return redirect('analysis:dataset')
            count = 0
            for _, row in df.iterrows():
                text = str(row['text']).strip()
                if text:
                    sentiment = determine_sentiment(text)
                    if not Dataset.objects.filter(text=text, sentiment=sentiment).exists():
                        Dataset.objects.create(text=text, sentiment=sentiment)
                        count += 1
            messages.success(request, f'Berhasil mengimpor {count} data dari Excel.')
        except Exception as e:
            logger.error(f"Error membaca Excel: {str(e)}")
            messages.error(request, f'Error membaca Excel: {str(e)}')
    return redirect('analysis:dataset')

# Views untuk Login dan Logout
def login_view(request):
    if request.user.is_authenticated:
        return redirect('analysis:dashboard')
    return render(request, 'analysis/login.html')

def login_action(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, 'Login berhasil.')
            return redirect('analysis:dashboard')
        else:
            messages.error(request, 'Username atau password salah.')
    return render(request, 'analysis/login.html')

def logout_view(request):
    logout(request)
    messages.success(request, "Anda telah berhasil logout.")
    return redirect('analysis:login')

# Views untuk Dashboard
def dashboard_view(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    if len(texts) < 6:
        logger.warning("Data tidak cukup untuk dashboard")
        return render(request, 'analysis/dashboard.html', {
            'nb_accuracy': 0,
            'svm_accuracy': 0,
            'error': 'Data tidak cukup (minimal 6 entri). Silakan tambahkan data di menu Dataset.'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = []
    valid_labels = []
    for text, label in zip(texts, labels):
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens) if tokens else ''
        if processed_text:
            processed_texts.append(processed_text)
            valid_labels.append(label)

    if len(processed_texts) < 6:
        logger.warning("Teks valid tidak cukup untuk dashboard")
        return render(request, 'analysis/dashboard.html', {
            'nb_accuracy': 0,
            'svm_accuracy': 0,
            'error': 'Teks valid tidak cukup setelah pemrosesan. Silakan tambahkan data di menu Dataset.'
        })

    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(processed_texts)
    y = valid_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    context = {
        'nb_accuracy': metrics["Naive Bayes"]['accuracy'],
        'svm_accuracy': metrics["SVM"]['accuracy'],
    }
    return render(request, 'analysis/dashboard.html', context)

# Views untuk TF-IDF
def tfidf_view(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]

    if not texts:
        logger.warning("Tidak ada data untuk TF-IDF")
        return render(request, 'analysis/tfidf.html', {
            'tfidf_data': [],
            'error': 'Tidak ada data tersedia. Silakan tambahkan data di menu Dataset.'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = []
    valid_texts = []
    for text in texts:
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens) if tokens else ''
        if processed_text:
            processed_texts.append(processed_text)
            valid_texts.append(text)

    if not processed_texts:
        logger.warning("Tidak ada teks valid untuk TF-IDF")
        return render(request, 'analysis/tfidf.html', {
            'tfidf_data': [],
            'error': 'Tidak ada teks valid setelah pemrosesan. Silakan tambahkan data di menu Dataset.'
        })

    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()

    tfidf_data = []
    for i, text in enumerate(valid_texts):
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        word_scores = [(feature_names[j], round(tfidf_scores[j], 4)) for j in range(len(feature_names)) if tfidf_scores[j] > 0]
        tfidf_data.append({'text': text, 'scores': word_scores})

    context = {'tfidf_data': tfidf_data}
    return render(request, 'analysis/tfidf.html', context)

# Views untuk Naive Bayes
def naive_bayes_dataset(request):
    dataset = Dataset.objects.all()
    search_response = request.GET.get('search_response', '')
    search_label = request.GET.get('search_label', '')

    if search_response:
        dataset = dataset.filter(text__icontains=search_response)
    if search_label:
        dataset = dataset.filter(sentiment__iexact=search_label)

    context = {
        'dataset': dataset,
        'search_response': search_response,
        'search_label': search_label,
        'active_menu': 'dataset',
    }
    return render(request, 'analysis/naive_bayes_dataset.html', context)

def naive_bayes_initial_process(request):
    logger.info("Mengakses naive_bayes_initial_process")
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]
    logger.debug(f"Jumlah entri dataset: {len(texts)}, Label: {labels}")

    if len(texts) < 6:
        logger.warning("Data tidak cukup untuk pemrosesan")
        return render(request, 'analysis/naive_bayes_initial_process.html', {
            'error': 'Data tidak cukup untuk pemrosesan (minimal 6 entri). Silakan tambahkan data di menu Dataset.',
            'active_menu': 'initial_process'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_data = []
    processed_texts = []
    valid_labels = []
    for text, label in zip(texts, labels):
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens) if tokens else ''
        if processed_text:
            processed_data.append((text, tokens))
            processed_texts.append(processed_text)
            valid_labels.append(label)
        logger.debug(f"Teks asli: {text}, Token: {tokens}")

    if len(processed_texts) < 6:
        logger.warning("Teks valid tidak cukup untuk pemrosesan")
        return render(request, 'analysis/naive_bayes_initial_process.html', {
            'error': 'Teks valid tidak cukup setelah pemrosesan (minimal 6 entri). Silakan tambahkan data di menu Dataset.',
            'active_menu': 'initial_process'
        })

    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(processed_texts)
    y = valid_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) * 100

    logger.info(f"Akurasi Naive Bayes: {accuracy}")
    context = {
        'processed_data': processed_data,
        'accuracy': round(accuracy, 2),
        'active_menu': 'initial_process'
    }
    return render(request, 'analysis/naive_bayes_initial_process.html', context)

def naive_bayes_performance(request):
    logger.info("Mengakses naive_bayes_performance")
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]
    logger.debug(f"Jumlah entri dataset: {len(texts)}, Label: {labels}")

    if len(texts) < 6:
        logger.warning("Data tidak cukup untuk evaluasi performa")
        return render(request, 'analysis/naive_bayes_performance.html', {
            'error': 'Data tidak cukup untuk evaluasi performa (minimal 6 entri). Silakan tambahkan data di menu Dataset.',
            'active_menu': 'performance'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = []
    valid_labels = []
    for text, label in zip(texts, labels):
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens) if tokens else ''
        if processed_text:
            processed_texts.append(processed_text)
            valid_labels.append(label)
        logger.debug(f"Teks asli: {text}, Teks diproses: {processed_text}")

    if len(processed_texts) < 6:
        logger.warning("Teks valid tidak cukup untuk evaluasi performa")
        return render(request, 'analysis/naive_bayes_performance.html', {
            'error': 'Teks valid tidak cukup setelah pemrosesan (minimal 6 entri). Silakan tambahkan data di menu Dataset.',
            'active_menu': 'performance'
        })

    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(processed_texts)
    y = valid_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="Naive Bayes")
    context = {
        'metrics': metrics["Naive Bayes"],
        'active_menu': 'performance'
    }
    return render(request, 'analysis/naive_bayes_performance.html', context)

def naive_bayes_prediksi(request):
    logger.info("Mengakses naive_bayes_prediksi")
    prediction = None
    input_text = None
    error = None

    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]
    logger.debug(f"Jumlah entri dataset: {len(texts)}, Label: {labels}")

    if len(texts) < 6:
        logger.warning("Data tidak cukup untuk pelatihan model")
        error = "Data tidak cukup untuk pelatihan model (minimal 6 entri). Silakan tambahkan data di menu Dataset."
        return render(request, 'analysis/naive_bayes_prediksi.html', {
            'error': error,
            'active_menu': 'prediksi'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = []
    valid_texts = []
    valid_labels = []
    for text, label in zip(texts, labels):
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens) if tokens else ''
        if processed_text:
            processed_texts.append(processed_text)
            valid_texts.append(text)
            valid_labels.append(label)
        logger.debug(f"Teks asli: {text}, Teks diproses: {processed_text}")

    if len(processed_texts) < 6:
        logger.warning("Teks valid tidak cukup untuk pelatihan model")
        error = "Teks valid tidak cukup setelah pemrosesan (minimal 6 entri). Silakan tambahkan data di menu Dataset."
        return render(request, 'analysis/naive_bayes_prediksi.html', {
            'error': error,
            'active_menu': 'prediksi'
        })

    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(processed_texts)
    y = valid_labels
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB(alpha=0.1)
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    model.fit(X_train_resampled, y_train_resampled)

    if request.method == 'POST':
        input_text = request.POST.get('text', '').strip()
        if input_text:
            tokens = preprocess_text(input_text)
            processed_input = ' '.join(tokens) if tokens else ''
            if processed_input:
                X_input = vectorizer.transform([processed_input])
                prediction = model.predict(X_input)[0]
                logger.info(f"Input: {input_text}, Prediksi: {prediction}")
            else:
                error = "Teks input tidak menghasilkan token valid setelah pemrosesan. Coba masukkan teks dengan kata-kata yang lebih jelas."
        else:
            error = "Silakan masukkan teks untuk prediksi."

    context = {
        'prediction': prediction,
        'input_text': input_text,
        'error': error,
        'active_menu': 'prediksi'
    }
    return render(request, 'analysis/naive_bayes_prediksi.html', context)

# Views untuk SVM
def svm_dataset(request):
    dataset = Dataset.objects.all()
    search_response = request.GET.get('search_response', '')
    search_label = request.GET.get('search_label', '')

    if search_response:
        dataset = dataset.filter(text__icontains=search_response)
    if search_label:
        dataset = dataset.filter(sentiment__iexact=search_label)

    context = {
        'dataset': dataset,
        'search_response': search_response,
        'search_label': search_label,
        'active_menu': 'dataset',
    }
    return render(request, 'analysis/svm_dataset.html', context)

def svm_initial_process(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]

    if len(texts) < 6:
        logger.warning("Data tidak cukup untuk pemrosesan SVM")
        return render(request, 'analysis/svm_initial_process.html', {
            'error': 'Data tidak cukup untuk pemrosesan (minimal 6 entri). Silakan tambahkan data di menu Dataset.',
            'active_menu': 'initial_process'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_data = []
    for text in texts:
        tokens = preprocess_text(text)
        processed_data.append((text, tokens))

    context = {
        'processed_data': processed_data,
        'active_menu': 'initial_process',
    }
    return render(request, 'analysis/svm_initial_process.html', context)

def svm_performance(request):
    logger.info("Mengakses svm_performance")
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]
    logger.debug(f"Jumlah entri dataset: {len(texts)}, Label: {labels}")

    if len(texts) < 6:
        logger.warning("Data tidak cukup untuk evaluasi performa SVM")
        return render(request, 'analysis/svm_performance.html', {
            'error': 'Data tidak cukup untuk evaluasi performa (minimal 6 entri). Silakan tambahkan data di menu Dataset.',
            'active_menu': 'performance'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = []
    valid_labels = []
    for text, label in zip(texts, labels):
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens) if tokens else ''
        if processed_text:
            processed_texts.append(processed_text)
            valid_labels.append(label)
        logger.debug(f"Teks asli: {text}, Teks diproses: {processed_text}")

    if len(processed_texts) < 6:
        logger.warning("Teks valid tidak cukup untuk evaluasi performa SVM")
        return render(request, 'analysis/svm_performance.html', {
            'error': 'Teks valid tidak cukup setelah pemrosesan (minimal 6 entri). Silakan tambahkan data di menu Dataset.',
            'active_menu': 'performance'
        })

    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(processed_texts)
    y = valid_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="SVM")
    context = {
        'metrics': metrics["SVM"],
        'active_menu': 'performance'
    }
    return render(request, 'analysis/svm_performance.html', context)

def svm_prediksi(request):
    logger.info("Mengakses svm_prediksi")
    prediction = None
    input_text = None
    error = None

    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]
    logger.debug(f"Jumlah entri dataset: {len(texts)}, Label: {labels}")

    if len(texts) < 6:
        logger.warning("Data tidak cukup untuk pelatihan model SVM")
        error = "Data tidak cukup untuk pelatihan model (minimal 6 entri). Silakan tambahkan data di menu Dataset."
        return render(request, 'analysis/svm_prediksi.html', {
            'error': error,
            'active_menu': 'prediksi'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = []
    valid_texts = []
    valid_labels = []
    for text, label in zip(texts, labels):
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens) if tokens else ''
        if processed_text:
            processed_texts.append(processed_text)
            valid_texts.append(text)
            valid_labels.append(label)
        logger.debug(f"Teks asli: {text}, Teks diproses: {processed_text}")

    if len(processed_texts) < 6:
        logger.warning("Teks valid tidak cukup untuk pelatihan model SVM")
        error = "Teks valid tidak cukup setelah pemrosesan (minimal 6 entri). Silakan tambahkan data di menu Dataset."
        return render(request, 'analysis/svm_prediksi.html', {
            'error': error,
            'active_menu': 'prediksi'
        })

    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(processed_texts)
    y = valid_labels
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearSVC(max_iter=1000)
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    model.fit(X_train_resampled, y_train_resampled)

    if request.method == 'POST':
        input_text = request.POST.get('text', '').strip()
        if input_text:
            tokens = preprocess_text(input_text)
            processed_input = ' '.join(tokens) if tokens else ''
            if processed_input:
                X_input = vectorizer.transform([processed_input])
                prediction = model.predict(X_input)[0]
                logger.info(f"Input: {input_text}, Prediksi: {prediction}")
            else:
                error = "Teks input tidak menghasilkan token valid setelah pemrosesan. Coba masukkan teks dengan kata-kata yang lebih jelas."
        else:
            error = "Silakan masukkan teks untuk prediksi."

    context = {
        'prediction': prediction,
        'input_text': input_text,
        'error': error,
        'active_menu': 'prediksi'
    }
    return render(request, 'analysis/svm_prediksi.html', context)

# Views untuk Summary Performance
def summary_performance(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    if len(texts) < 6:
        logger.warning("Data tidak cukup untuk summary performance")
        return render(request, 'analysis/summary_performance.html', {
            'nb_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
            'svm_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
            'nb_confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'svm_confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'error': 'Data tidak cukup (minimal 6 entri). Silakan tambahkan data di menu Dataset.'
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = []
    valid_labels = []
    for text, label in zip(texts, labels):
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens) if tokens else ''
        if processed_text:
            processed_texts.append(processed_text)
            valid_labels.append(label)

    if len(processed_texts) < 6:
        logger.warning("Teks valid tidak cukup untuk summary performance")
        return render(request, 'analysis/summary_performance.html', {
            'nb_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
            'svm_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
            'nb_confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'svm_confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'error': 'Teks valid tidak cukup setelah pemrosesan. Silakan tambahkan data di menu Dataset.'
        })

    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(processed_texts)
    y = valid_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics, predictions = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    nb_cm = confusion_matrix(y_test, predictions["Naive Bayes"], labels=['positif', 'negatif', 'netral'])
    svm_cm = confusion_matrix(y_test, predictions["SVM"], labels=['positif', 'negatif', 'netral'])

    context = {
        'nb_metrics': metrics["Naive Bayes"],
        'svm_metrics': metrics["SVM"],
        'nb_confusion_matrix': nb_cm.tolist(),
        'svm_confusion_matrix': svm_cm.tolist(),
    }
    return render(request, 'analysis/summary_performance.html', context)

# Views untuk Manajemen User
def admin_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.error(request, "Silakan login terlebih dahulu.")
            return redirect('analysis:login')
        if not request.user.is_superuser:
            messages.error(request, "Hanya admin yang dapat mengakses halaman ini.")
            return redirect('analysis:dashboard')
        return view_func(request, *args, **kwargs)
    return wrapper

@login_required
@admin_required
def manajemen_user(request):
    users = User.objects.all()
    search_username = request.GET.get('search_username', '')
    if search_username:
        users = users.filter(username__icontains=search_username)
    return render(request, 'analysis/manajemen_user.html', {
        'users': users,
        'search_username': search_username
    })

@login_required
@admin_required
def add_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        is_active = request.POST.get('is_active') == 'on'

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username sudah digunakan.")
            return redirect('analysis:add_user')
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email sudah digunakan.")
            return redirect('analysis:add_user')

        User.objects.create_user(
            username=username,
            email=email,
            password=password,
            is_active=is_active
        )
        messages.success(request, "Pengguna berhasil ditambahkan.")
        return redirect('analysis:manajemen_user')
    return render(request, 'analysis/add_user.html')

@login_required
@admin_required
def edit_user(request, id):
    user = get_object_or_404(User, id=id)
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        is_active = request.POST.get('is_active') == 'on'

        if username != user.username and User.objects.filter(username=username).exists():
            messages.error(request, "Username sudah digunakan.")
            return redirect('analysis:edit_user', id=id)
        if email != user.email and User.objects.filter(email=email).exists():
            messages.error(request, "Email sudah digunakan.")
            return redirect('analysis:edit_user', id=id)

        user.username = username
        user.email = email
        if password:
            user.set_password(password)
        user.is_active = is_active
        user.save()
        messages.success(request, "Pengguna berhasil diperbarui.")
        return redirect('analysis:manajemen_user')
    return render(request, 'analysis/edit_user.html', {'user': user})

@login_required
@admin_required
def delete_user(request, id):
    user = get_object_or_404(User, id=id)
    if user == request.user:
        messages.error(request, "Anda tidak dapat menghapus akun Anda sendiri.")
        return redirect('analysis:manajemen_user')
    user.delete()
    messages.success(request, "Pengguna berhasil dihapus.")
    return redirect('analysis:manajemen_user')