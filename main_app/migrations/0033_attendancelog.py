# Generated by Django 5.0.6 on 2024-07-03 08:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0032_auto_20240702_1922'),
    ]

    operations = [
        migrations.CreateModel(
            name='AttendanceLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('check_in_time', models.DateTimeField()),
                ('check_out_time', models.DateTimeField(blank=True, null=True)),
                ('duration', models.DurationField(blank=True, null=True)),
                ('is_approved', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('guard', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main_app.guard')),
            ],
        ),
    ]
