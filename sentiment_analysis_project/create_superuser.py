import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_analysis_project.settings')
django.setup()

from django.contrib.auth.models import User

def create_superuser():
    User.objects.create_superuser(
        username='admin',
        email='admin@example.com',
        password='Admin123!'
    )
    print("Superuser created successfully!")

if __name__ == "__main__":
    create_superuser()