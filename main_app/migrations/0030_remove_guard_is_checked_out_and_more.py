# Generated by Django 5.0.6 on 2024-06-30 11:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0029_guard_is_checked_out_guardlocation_check_in_time_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='guard',
            name='is_checked_out',
        ),
        migrations.RemoveField(
            model_name='guardlocation',
            name='check_in_time',
        ),
        migrations.RemoveField(
            model_name='guardlocation',
            name='check_out_time',
        ),
    ]
