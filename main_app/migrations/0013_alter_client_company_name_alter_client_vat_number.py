# Generated by Django 5.0.6 on 2024-06-06 07:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0012_client_company_name_client_vat_number'),
    ]

    operations = [
        migrations.AlterField(
            model_name='client',
            name='company_name',
            field=models.CharField(default='', max_length=255),
        ),
        migrations.AlterField(
            model_name='client',
            name='vat_number',
            field=models.CharField(default='', max_length=255),
        ),
    ]
