# Generated by Django 5.0.6 on 2024-06-29 10:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0026_guardreview_updated_at_alter_guardreview_rating_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='site',
            name='latitude',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='site',
            name='longitude',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
