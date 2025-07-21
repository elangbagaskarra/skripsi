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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi pra-pemrosesan teks (digunakan di semua view)
def preprocess_text(text, stop_words):
    tokens = word_tokenize(text.lower())
    return ' '.join([t for t in tokens if t not in stop_words and t.isalpha()])

# Fungsi pelatihan dan evaluasi model (digunakan di semua view)
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="both"):
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC()
    }
    metrics = {}
    predictions = {}
    for name, model in models.items():
        if model_type != "both" and name != model_type:
            continue
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        metrics[name] = {
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
            'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        }
    return metrics, predictions

# Login
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
            return redirect('analysis:dashboard')  # Arahkan ke dashboard setelah login
        else:
            return render(request, 'analysis/login.html', {'error': 'Username atau password salah'})
    return render(request, 'analysis/login.html')

# Logout
def logout_view(request):
    logout(request)
    messages.success(request, "Anda telah berhasil logout.")
    return redirect('analysis:login')

# Dashboard (tanpa login)
def dashboard_view(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    if len(texts) < 2:
        return render(request, 'analysis/dashboard.html', {'nb_accuracy': 0, 'svm_accuracy': 0})

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = [preprocess_text(text, stop_words) for text in texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluasi model
    metrics, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    context = {
        'nb_accuracy': metrics["Naive Bayes"]['accuracy'],
        'svm_accuracy': metrics["SVM"]['accuracy'],
    }
    return render(request, 'analysis/dashboard.html', context)

# Dataset (tanpa login)
def dataset_view(request):
    dataset = Dataset.objects.all()
    search_text = request.GET.get('search_text', '')
    search_sentiment = request.GET.get('search_sentiment', '')

    if search_text:
        dataset = dataset.filter(text__icontains=search_text)
    if search_sentiment:
        dataset = dataset.filter(sentiment__iexact=search_sentiment)

    context = {
        'dataset': dataset,
        'search_text': search_text,
        'search_sentiment': search_sentiment,
    }
    return render(request, 'analysis/dataset.html', context)

def add_data(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        sentiment = request.POST.get('sentiment')
        Dataset.objects.create(text=text, sentiment=sentiment)
        messages.success(request, "Data berhasil ditambahkan.")
        return redirect('analysis:dataset')
    return render(request, 'analysis/dataset.html')

def edit_data(request, id):
    data = get_object_or_404(Dataset, id=id)
    if request.method == 'POST':
        data.text = request.POST.get('text')
        data.sentiment = request.POST.get(' sentiment')
        data.save()
        messages.success(request, "Data berhasil diperbarui.")
        return redirect('analysis:dataset')
    context = {'data': data}
    return render(request, 'analysis/dataset.html', context)

def delete_data(request, id):
    data = get_object_or_404(Dataset, id=id)
    data.delete()
    messages.success(request, "Data berhasil dihapus.")
    return redirect('analysis:dataset')

def upload_excel(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        excel_file = request.FILES['excel_file']
        try:
            df = pd.read_excel(excel_file)
            for _, row in df.iterrows():
                Dataset.objects.create(
                    text=row['text'],
                    sentiment=row['sentiment']
                )
            messages.success(request, "Data dari Excel berhasil diunggah.")
        except Exception as e:
            messages.error(request, f"Terjadi kesalahan: {str(e)}")
    return redirect('analysis:dataset')

# TF-IDF (tanpa login)
def tfidf_view(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]

    if not texts:
        return render(request, 'analysis/tfidf.html', {'tfidf_data': []})

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = [preprocess_text(text, stop_words) for text in texts]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_data = []
    for i in range(len(texts)):
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        word_scores = [(feature_names[j], round(tfidf_scores[j], 4)) for j in range(len(feature_names)) if tfidf_scores[j] > 0]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        tfidf_data.append({'text': texts[i], 'scores': word_scores[:5]})

    context = {'tfidf_data': tfidf_data}
    return render(request, 'analysis/tfidf.html', context)

# Naive Bayes (tanpa login)
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
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]

    stop_words = set(stopwords.words('indonesian'))
    processed_data = [(text, preprocess_text(text, stop_words).split()) for text in texts]
    context = {
        'processed_data': processed_data,
        'active_menu': 'initial_process',
    }
    return render(request, 'analysis/naive_bayes_initial_process.html', context)

def naive_bayes_performance(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    if len(texts) < 2:
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0}
        return render(request, 'analysis/naive_bayes_performance.html', {
            'metrics': metrics,
            'active_menu': 'performance',
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = [preprocess_text(text, stop_words) for text in texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="Naive Bayes")
    context = {
        'metrics': metrics["Naive Bayes"],
        'active_menu': 'performance',
    }
    return render(request, 'analysis/naive_bayes_performance.html', context)

def naive_bayes_prediksi(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = [preprocess_text(text, stop_words) for text in texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    y = labels

    nb_model = MultinomialNB()
    nb_model.fit(X, y)

    prediction = None
    input_text = ''
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        if input_text:
            processed_input = preprocess_text(input_text, stop_words)
            X_input = vectorizer.transform([processed_input])
            prediction = nb_model.predict(X_input)[0]

    context = {
        'prediction': prediction,
        'input_text': input_text,
        'active_menu': 'prediksi',
    }
    return render(request, 'analysis/naive_bayes_prediksi.html', context)

# SVM (tanpa login)
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

    stop_words = set(stopwords.words('indonesian'))
    processed_data = [(text, preprocess_text(text, stop_words).split()) for text in texts]
    context = {
        'processed_data': processed_data,
        'active_menu': 'initial_process',
    }
    return render(request, 'analysis/svm_initial_process.html', context)

def svm_performance(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    if len(texts) < 2:
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0}
        return render(request, 'analysis/svm_performance.html', {
            'metrics': metrics,
            'active_menu': 'performance',
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = [preprocess_text(text, stop_words) for text in texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="SVM")
    context = {
        'metrics': metrics["SVM"],
        'active_menu': 'performance',
    }
    return render(request, 'analysis/svm_performance.html', context)

def svm_prediksi(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = [preprocess_text(text, stop_words) for text in texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    y = labels

    svm_model = LinearSVC()
    svm_model.fit(X, y)

    prediction = None
    input_text = ''
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        if input_text:
            processed_input = preprocess_text(input_text, stop_words)
            X_input = vectorizer.transform([processed_input])
            prediction = svm_model.predict(X_input)[0]

    context = {
        'prediction': prediction,
        'input_text': input_text,
        'active_menu': 'prediksi',
    }
    return render(request, 'analysis/svm_prediksi.html', context)

# Summary Performance (tanpa login)
def summary_performance(request):
    dataset = Dataset.objects.all()
    texts = [d.text for d in dataset]
    labels = [d.sentiment for d in dataset]

    if len(texts) < 2:
        return render(request, 'analysis/summary_performance.html', {
            'nb_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0},
            'svm_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0},
            'nb_confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'svm_confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        })

    stop_words = set(stopwords.words('indonesian'))
    processed_texts = [preprocess_text(text, stop_words) for text in texts]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    y = labels
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

# Manajemen User (dengan login dan admin)
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

    context = {
        'users': users,
        'search_username': search_username,
    }
    return render(request, 'analysis/manajemen_user.html', context)

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

        user = User(
            username=username,
            email=email,
            password=make_password(password),
            is_active=is_active
        )
        user.save()
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
            user.password = make_password(password)
        user.is_active = is_active
        user.save()
        messages.success(request, "Pengguna berhasil diperbarui.")
        return redirect('analysis:manajemen_user')

    context = {'user': user}
    return render(request, 'analysis/edit_user.html', context)

@login_required
@admin_required
def delete_user(request, id):
    user = get_object_or_404(User, id=id)
    if user == request.user:
        messages.error(request, "Anda tidak dapat menghapus akun Anda sendiri.")
        return redirect('analysis:manajemen_user')
    user.delete()
    messages.success(request, "Pengguna berhasil dihapus.")  # Memperbaiki typo messagesA
    return redirect('analysis:manajemen_user')