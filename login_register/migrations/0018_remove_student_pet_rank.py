# Generated by Django 4.1.9 on 2023-06-25 20:07

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('login_register', '0017_pet_pet_rank'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='student',
            name='pet_rank',
        ),
    ]