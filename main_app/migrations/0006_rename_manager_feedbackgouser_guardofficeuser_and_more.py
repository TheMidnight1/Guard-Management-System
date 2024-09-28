# Generated by Django 5.0.6 on 2024-06-03 06:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main_app', '0005_rename_division_guard_guard_office_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='feedbackgouser',
            old_name='manager',
            new_name='guardofficeuser',
        ),
        migrations.RenameField(
            model_name='leavereportgouser',
            old_name='manager',
            new_name='guardofficeuser',
        ),
        migrations.RenameField(
            model_name='notificationgouser',
            old_name='manager',
            new_name='guardofficeuser',
        ),
        migrations.AlterField(
            model_name='customuser',
            name='user_type',
            field=models.CharField(choices=[(1, 'CEO'), (2, 'GuardOfficeUser'), (3, 'Employee')], default=1, max_length=1),
        ),
    ]
