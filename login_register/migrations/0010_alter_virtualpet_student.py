# Generated by Django 4.1.9 on 2023-06-24 22:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('login_register', '0009_alter_virtualpet_pet_name_alter_virtualpet_pet_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='virtualpet',
            name='student',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='pets', to='login_register.student'),
        ),
    ]