# Generated by Django 5.0.6 on 2024-06-28 12:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0024_guardreview'),
    ]

    operations = [
        migrations.AddField(
            model_name='guard',
            name='average_rating',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='guard',
            name='review_count',
            field=models.IntegerField(default=0),
        ),
    ]
