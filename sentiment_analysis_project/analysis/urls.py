from django.urls import path
from . import views

app_name = 'analysis'  # Namespace untuk aplikasi analysis

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('dashboard/', views.dashboard_view, name='dashboard_explicit'),  # Path baru untuk dashboard
    path('login/', views.login_view, name='login'),
path('login_action/', views.login_action, name='login_action'),
    path('logout/', views.logout_view, name='logout'),
    path('dataset/', views.dataset_view, name='dataset'),
    path('dataset/add/', views.add_data, name='add_data'),
    path('dataset/edit/<int:id>/', views.edit_data, name='edit_data'),
    path('dataset/delete/<int:id>/', views.delete_data, name='delete_data'),
    path('dataset/upload/', views.upload_excel, name='upload_excel'),
    path('tfidf/', views.tfidf_view, name='tfidf'),
    path('naive-bayes/', views.naive_bayes_dataset, name='naive_bayes_dataset'),
    path('naive-bayes/initial-process/', views.naive_bayes_initial_process, name='naive_bayes_initial_process'),
    path('naive-bayes/performance/', views.naive_bayes_performance, name='naive_bayes_performance'),
    path('naive-bayes/prediksi/', views.naive_bayes_prediksi, name='naive_bayes_prediksi'),
    path('svm/', views.svm_dataset, name='svm_dataset'),
    path('svm/initial-process/', views.svm_initial_process, name='svm_initial_process'),
    path('svm/performance/', views.svm_performance, name='svm_performance'),
    path('svm/prediksi/', views.svm_prediksi, name='svm_prediksi'),
    path('summary-performance/', views.summary_performance, name='summary_performance'),
    path('manajemen-user/', views.manajemen_user, name='manajemen_user'),
    path('manajemen-user/add/', views.add_user, name='add_user'),
    path('manajemen-user/edit/<int:id>/', views.edit_user, name='edit_user'),
    path('manajemen-user/delete/<int:id>/', views.delete_user, name='delete_user'),
]