# Generated by Django 5.0.6 on 2024-06-09 09:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0017_complaint'),
    ]

    operations = [
        migrations.AddField(
            model_name='notificationgouser',
            name='is_complaint',
            field=models.BooleanField(default=False),
        ),
    ]
