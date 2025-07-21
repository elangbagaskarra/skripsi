import os
import shutil
import string
import logging
import json
import subprocess
import tempfile
import pandas as pd
import nltk
import re
from collections import Counter
from wordcloud import WordCloud
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from .models import Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from io import BytesIO
import base64
from .models import Dataset
from .utils import case_folding, cleansing, tokenizing, stemming, stopword_removal,load_stopwords_from_file

import os
from django.conf import settings 



# Setup logging
logger = logging.getLogger(__name__)

# Inisialisasi stemmer dan stopword Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

# Load lexicon dan kamus slang/stop words
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEXICON_PATH = os.path.join(BASE_DIR, 'lexicon', 'modified_full_lexicon.csv')
SLANG_PATH = os.path.join(BASE_DIR, 'lexicon', 'slang.csv')
STOPWORD_PATH = os.path.join(BASE_DIR, 'lexicon', 'stopword.txt')

def remove_urls(text):
    # Regex untuk mendeteksi URL (termasuk https://t.co)
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)
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
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "SVM": LinearSVC(max_iter=1000)
    }
    metrics = {}
    predictions = {}
    for name, model in models.items():
        if model_type != "both" and name != model_type:
            continue
        try:
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
                Dataset.objects.create(text=text, sentiment=sentiment)
                messages.success(request, 'Data berhasil ditambahkan.')
            else:
                messages.error(request, 'Teks tidak valid setelah stemming.')
        else:
            messages.error(request, 'Data tidak valid.')
    return redirect('analysis:dataset')

def edit_data(request, id):
    data = get_object_or_404(Dataset, id=id)
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        sentiment = request.POST.get('sentiment', '').strip()
        if text and sentiment in ['positif', 'negatif', 'netral']:
            data.text = text
            data.sentiment = sentiment
            data.save()
            messages.success(request, 'Data berhasil diperbarui.')
        else:
            messages.error(request, 'Data tidak valid.')
    return redirect('analysis:dataset')

def delete_data(request, id):
    data = get_object_or_404(Dataset, id=id)
    data.delete()
    messages.success(request, 'Data berhasil dihapus.')
    return redirect('analysis:dataset')

def crawl_twitter(request):
    if request.method == 'POST':
        keyword = request.POST.get('twitter_keyword', '').strip()
        try:
            tweet_limit = int(request.POST.get('tweet_limit', 10))
            tweet_limit = min(tweet_limit, 50)  # Batasi untuk stabilitas
        except ValueError:
            messages.error(request, 'Batas tweet harus berupa angka.')
            return redirect('analysis:dataset')
        
        if not keyword:
            messages.error(request, 'Kata kunci tidak boleh kosong.')
            return redirect('analysis:dataset')
        
        try:
            # Bersihkan kata kunci untuk nama file
            safe_keyword = re.sub(r'[^a-zA-Z0-9_-]', '', keyword.replace(' ', '_'))
            if not safe_keyword:
                safe_keyword = 'tweets'  # Fallback
            
            # Memastikan directory exists dan nama file aman
            tweets_data_dir = os.path.join(BASE_DIR, 'tweets-data')
            os.makedirs(tweets_data_dir, exist_ok=True)
            output_file = os.path.join(tweets_data_dir, f"{safe_keyword}.csv")
            
            tweet_harvest_path = os.path.join(BASE_DIR, 'tweet-harvest')
            
            # Validasi folder tweet-harvest
            if not os.path.exists(tweet_harvest_path):
                raise FileNotFoundError(f"Folder tweet-harvest tidak ditemukan di {tweet_harvest_path}")
            
            # Set environment variable for auth token
            auth_token = os.getenv('TWITTER_AUTH_TOKEN', '0c0be3298e17427f6fd02f3c468e5d479e841514')
            
            # Prepare the command as a list - much safer than using shell=True
            cmd = [
                'npx',
                'tweet-harvest',
                '-s', keyword,
                '-l', str(tweet_limit),
                '-t', auth_token,
                '-o', output_file,
                '--delay', '5000'
            ]
            
            logger.debug(f"Menjalankan perintah: {' '.join(cmd)}")
            
            # Run the command without shell=True for security
            process = subprocess.Popen(
                cmd,
                cwd=tweet_harvest_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=120)  # Add timeout
            
            if process.returncode != 0:
                logger.error(f"Proses gagal dengan kode: {process.returncode}")
                logger.error(f"stderr: {stderr}")
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr)
            
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"File output {output_file} tidak dibuat.")
            
            # Process CSV file
            df = pd.read_csv(output_file)
            count = 0
            for _, row in df.iterrows():
                text = str(row.get('text', '')).strip()
                if text and not Dataset.objects.filter(text=text).exists():
                    sentiment = determine_sentiment(text)
                    Dataset.objects.create(text=text, sentiment=sentiment)
                    count += 1
            
            if count > 0:
                messages.success(request, f'Berhasil mengambil {count} tweet untuk kata kunci "{keyword}".')
            else:
                messages.warning(request, f'Tidak ada tweet baru ditemukan untuk kata kunci "{keyword}".')
            
            return redirect('analysis:dataset')
        
        except FileNotFoundError as e:
            logger.error(f"Kesalahan file: {str(e)}")
            messages.error(request, f'Gagal mengambil data Twitter: {str(e)}')
        except subprocess.CalledProcessError as e:
            logger.error(f"Gagal menjalankan Tweet Harvest: {e}")
            logger.error(f"Stderr: {e.stderr}")
            messages.error(request, f'Gagal mengambil data Twitter: Perintah tidak dapat dijalankan')
        except subprocess.TimeoutExpired as e:
            logger.error(f"Waktu habis saat menjalankan Tweet Harvest: {e}")
            messages.error(request, 'Waktu pemrosesan habis. Coba gunakan batas tweet yang lebih kecil.')
        except Exception as e:
            logger.error(f"Gagal mengambil data Twitter: {str(e)}")
            messages.error(request, f'Gagal mengambil data Twitter: {str(e)}')
        
    return redirect('analysis:dataset')

# Views untuk Instagram Scraper

def crawl_instagram(request):
    if request.method == 'POST':
        hashtag = request.POST.get('instagram_hashtag', '').strip()

        # Validasi input hashtag
        if not hashtag:
            messages.error(request, 'Hashtag tidak boleh kosong.')
            return redirect('analysis:dataset')
        
        try:
            post_limit = int(request.POST.get('post_limit', 10))
            post_limit = min(post_limit, 50)  # Batasi maksimal 50 postingan
        except ValueError:
            messages.error(request, 'Batas postingan harus berupa angka.')
            return redirect('analysis:dataset')

        # Remove # if it was included in the hashtag
        if hashtag.startswith('#'):
            hashtag = hashtag[1:]

        try:
            # Buat direktori sementara untuk menyimpan file hasil scraping
            output_dir = tempfile.mkdtemp()
            
            # Jalankan perintah Instagram Scraper
            command = [
                "instagram-scraper",
                hashtag,
                "--tag",
                "--maximum", str(post_limit),
                "--destination", output_dir,
                "--media-metadata",
                "--media-types", "none"
            ]
            
            logger.debug(f"Menjalankan perintah Instagram: {' '.join(command)}")
            
            # Run command with proper subprocess handling
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=120)  # Add timeout
            
            output_file = os.path.join(output_dir, f"{hashtag}.json")
            
            if process.returncode != 0:
                logger.error(f"Instagram scraper gagal dengan kode: {process.returncode}")
                logger.error(f"Stderr: {stderr}")
                raise subprocess.CalledProcessError(process.returncode, command, stderr)
            
            # Check if the file exists
            if not os.path.exists(output_file):
                logger.warning(f"File output {output_file} tidak ditemukan.")
                messages.warning(request, f'Tidak ditemukan data untuk hashtag #{hashtag}.')
                return redirect('analysis:dataset')
            
            # Process the JSON file
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    posts_data = json.load(f)
                
                count = 0
                for post in posts_data:
                    # Instagram API structure can change, handle with care
                    try:
                        edges = post.get('edge_media_to_caption', {}).get('edges', [])
                        if edges and len(edges) > 0:
                            text = edges[0].get('node', {}).get('text', '')
                            if text and len(text.strip()) > 0:
                                # Check if text already exists in dataset
                                if not Dataset.objects.filter(text=text).exists():
                                    sentiment = determine_sentiment(text)
                                    Dataset.objects.create(text=text, sentiment=sentiment)
                                    count += 1
                    except Exception as e:
                        logger.error(f"Error processing post: {str(e)}")
                
                if count > 0:
                    messages.success(request, f'Berhasil mengambil {count} postingan dari Instagram untuk hashtag #{hashtag}.')
                else:
                    messages.warning(request, f'Tidak ada postingan baru ditemukan untuk hashtag #{hashtag}.')
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {str(e)}")
                messages.error(request, f'Error membaca data JSON: {str(e)}')
            
        except FileNotFoundError as e:
            logger.error(f"Program instagram-scraper tidak ditemukan: {str(e)}")
            messages.error(request, 'Program instagram-scraper tidak ditemukan. Pastikan sudah terinstall dengan "pip install instagram-scraper".')
        except subprocess.CalledProcessError as e:
            logger.error(f"Gagal menjalankan Instagram Scraper: {e}")
            messages.error(request, f'Gagal menjalankan Instagram Scraper. Periksa kembali hashtag yang digunakan.')
        except subprocess.TimeoutExpired as e:
            logger.error(f"Waktu habis saat menjalankan Instagram Scraper: {e}")
            messages.error(request, 'Waktu pemrosesan habis. Coba gunakan batas postingan yang lebih kecil.')
        except Exception as e:
            logger.error(f"Gagal mengambil data Instagram: {str(e)}")
            messages.error(request, f'Gagal mengambil data Instagram: {str(e)}')
        
        finally:
            # Hapus direktori sementara jika sudah selesai
            if 'output_dir' in locals() and os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
        
    return redirect('analysis:dataset')
def upload_excel(request):
    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']
        try:
            # Validasi ekstensi file
            if not csv_file.name.endswith('.csv'):
                messages.error(request, "File harus memiliki ekstensi .csv.")
                return redirect('analysis:dataset')

            # Validasi ukuran file
            max_size = 10 * 1024 * 1024  # 10MB
            if csv_file.size > max_size:
                messages.error(request, "Ukuran file terlalu besar. Maksimum 10MB.")
                return redirect('analysis:dataset')

            # Logging informasi file (opsional)
            logger.debug(f"Mengunggah file CSV: {csv_file.name}, Ukuran: {csv_file.size} bytes")

            # Coba membaca file CSV dengan beberapa encoding
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            df = None
            for encoding in encodings:
                try:
                    csv_file.seek(0)
                    df = pd.read_csv(csv_file, encoding=encoding)
                    logger.debug(f"Berhasil membaca CSV dengan encoding: {encoding}")
                    break
                except Exception as e:
                    logger.warning(f"Gagal membaca CSV dengan encoding {encoding}: {str(e)}")
                    continue

            if df is None:
                raise ValueError("Gagal membaca file CSV dengan encoding yang didukung (utf-8, latin1, iso-8859-1).")

            # --- Cari seluruh kolom yang bisa dipakai untuk teks/caption ---
            text_columns = []
            for col in df.columns:
                if (
                    col.lower() == 'text'
                    or col.lower() == 'full_text'
                    or col.lower() == 'caption'
                    or (col.lower().endswith('/caption'))
                ):
                    text_columns.append(col)

            if not text_columns:
                messages.error(request, "File CSV harus memiliki setidaknya satu kolom: 'text', 'full_text', atau 'caption' (boleh juga 'topPosts/x/caption').")
                return redirect('analysis:dataset')

            count = 0
            for _, row in df.iterrows():
                for col in text_columns:
                    value = row.get(col)
                    text = str(value).strip() if value is not None else ""
                    if text and text.lower() != 'nan':
                        sentiment = determine_sentiment(text)
                        # Cek duplikat berdasarkan text dan sentiment
                        if not Dataset.objects.filter(text=text, sentiment=sentiment).exists():
                            Dataset.objects.create(text=text, sentiment=sentiment)
                            count += 1
            messages.success(request, f'Berhasil mengimpor {count} data dari CSV.')
        except ValueError as e:
            logger.error(f"Error membaca CSV: {str(e)}")
            messages.error(request, f'Error membaca CSV: {str(e)}')
        except Exception as e:
            logger.error(f"Error tak terduga saat membaca CSV: {str(e)}")
            messages.error(request, f'Error tak terduga saat membaca CSV: {str(e)}')
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

def get_three_word_combinations(texts):
    combinations = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) >= 3:
            combinations.extend([tuple(words[i:i+3]) for i in range(len(words)-2)])
    return combinations

def dashboard_view(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    # Validasi jika dataset tidak cukup
    if len(texts) < 6:
        return render(request, 'analysis/dashboard.html', {
            'error': 'Data tidak cukup (minimal 6 entri). Silakan tambahkan data di menu Dataset.'
        })

    # Menghitung statistik sentimen
    stats = {
        'total_dataset': len(dataset),
        'positif': dataset.filter(sentiment='positif').count(),
        'negatif': dataset.filter(sentiment='negatif').count(),
        'netral': dataset.filter(sentiment='netral').count()
    }

    # Menghitung jumlah kata dalam dataset
    total_kata = {
        'total_kata': sum(len(text.split()) for text in texts),
        'kata_positif': sum(len(text.split()) for text, sentiment in zip(texts, labels) if sentiment == 'positif'),
        'kata_negatif': sum(len(text.split()) for text, sentiment in zip(texts, labels) if sentiment == 'negatif'),
        'kata_netral': sum(len(text.split()) for text, sentiment in zip(texts, labels) if sentiment == 'netral')
    }

    # TF-IDF untuk fitur teks
    vectorizer = TfidfVectorizer(max_features=500, min_df=1, max_df=0.8)
    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih dan mengevaluasi model
    metrics = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(max_iter=1000)
    }
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics[model_name] = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            'f1': round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        }

    # Top 10 Words in Each Sentiment
    top_words = {
        'positif': Counter(' '.join([text for text, sentiment in zip(texts, labels) if sentiment == 'positif']).split()).most_common(10),
        'negatif': Counter(' '.join([text for text, sentiment in zip(texts, labels) if sentiment == 'negatif']).split()).most_common(10),
        'netral': Counter(' '.join([text for text, sentiment in zip(texts, labels) if sentiment == 'netral']).split()).most_common(10)
    }

    # Top 10 Three Word Combinations
    three_word_combinations = get_three_word_combinations(texts)
    top_three_word_combinations = Counter(three_word_combinations).most_common(10)

    # Word Cloud dengan pembersihan URL
    cleaned_texts = [remove_urls(text) for text in texts]
    word_cloud_data = ' '.join(cleaned_texts)
    img_base64 = None
    if word_cloud_data.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(word_cloud_data)
        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    else:
        logger.warning("Tidak ada teks valid untuk membuat word cloud setelah pembersihan URL.")

    context = {
        'stats': stats,
        'total_kata': total_kata,
        'metrics': metrics,
        'top_words': top_words,
        'top_three_word_combinations': top_three_word_combinations,
        'wordcloud_image': img_base64
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
        input_text = request.POST.get('input_text', '').strip()
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


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(max_iter=1000),
    }
    metrics = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
        }
    return metrics, predictions

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



def manajemen_user(request):
    users = User.objects.all()
    search_username = request.GET.get('search_username', '')
    if search_username:
        users = users.filter(username__icontains=search_username)
    return render(request, 'analysis/manajemen_user.html', {
        'users': users,
        'search_username': search_username
    })


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


def delete_user(request, id):
    user = get_object_or_404(User, id=id)
    if user == request.user:
        messages.error(request, "Anda tidak dapat menghapus akun Anda sendiri.")
        return redirect('analysis:manajemen_user')
    user.delete()
    messages.success(request, "Pengguna berhasil dihapus.")
    return redirect('analysis:manajemen_user')


def preprocessing_view(request):
    # Ambil 5 data teratas dari dataset
    data = Dataset.objects.all()[:5]

    # Ambil path absolut ke stopword.txt
    stopword_file_path = os.path.join(settings.BASE_DIR, 'analysis', 'lexicon', 'stopword.txt')

    # Load stopword dari file teks (hasil: set dalam lowercase)
    stopword_list = load_stopwords_from_file(stopword_file_path)

    result_list = []

    for item in data:
        original_text = item.text

        folded   = case_folding(original_text)
        cleaned  = cleansing(folded)
        tokens   = tokenizing(cleaned)
        removed  = stopword_removal(tokens, stopword_list)
        stemmed  = stemming(removed)

        result_list.append({
            'original': original_text,
            'case_folding': folded,
            'cleansing': cleaned,
            'tokenizing': tokens,
            'stopword_removal': removed,
            'stemming': stemmed,
        })

    return render(request, 'analysis/preprocessing.html', {
        'results': result_list
    })
def case_folding_view(request):
    data = Dataset.objects.all()  # Ambil semua data tanpa limit
    result = [{'original': d.text, 'case_folding': case_folding(d.text)} for d in data]
    return render(request, 'analysis/case_folding.html', {'results': result, 'active_menu': 'case_folding'})


def cleansing_view(request):
    data = Dataset.objects.all() 
    result = [{'original': d.text, 'cleansing': cleansing(case_folding(d.text))} for d in data]
    return render(request, 'analysis/cleansing.html', {'results': result, 'active_menu': 'cleansing'})

def tokenizing_view(request):
    data = Dataset.objects.all() 
    result = [{'original': d.text, 'tokenizing': tokenizing(cleansing(case_folding(d.text)))} for d in data]
    return render(request, 'analysis/tokenizing.html', {'results': result, 'active_menu': 'tokenizing'})

def stopword_view(request):
    stopwords = load_stopwords_from_file(os.path.join(settings.BASE_DIR, 'analysis', 'lexicon', 'stopword.txt'))
    data = Dataset.objects.all() 
    result = [{
        'original': d.text,
        'stopword_removal': stopword_removal(tokenizing(cleansing(case_folding(d.text))), stopwords)
    } for d in data]
    return render(request, 'analysis/stopword.html', {'results': result, 'active_menu': 'stopword'})

def stemming_view(request):
    stopwords = load_stopwords_from_file(os.path.join(settings.BASE_DIR, 'analysis', 'lexicon', 'stopword.txt'))
    data = Dataset.objects.all() 
    result = [{
        'original': d.text,
        'stemming': stemming(stopword_removal(tokenizing(cleansing(case_folding(d.text))), stopwords))
    } for d in data]
    return render(request, 'analysis/stemming.html', {'results': result, 'active_menu': 'stemming'})
