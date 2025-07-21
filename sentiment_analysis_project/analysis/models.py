from django.db import models

class Dataset(models.Model):
    text = models.TextField()
    sentiment = models.CharField(max_length=10, choices=[
        ('positif', 'Positif'),
        ('negatif', 'Negatif'),
        ('netral', 'Netral'),
    ])

    def __str__(self):
        return f"{self.text[:50]} - {self.sentiment}"