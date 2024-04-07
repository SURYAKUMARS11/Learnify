# Generated by Django 4.1.9 on 2023-06-25 19:18

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('login_register', '0013_alter_virtualpet_pet_name_alter_virtualpet_pet_type_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Pet',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pet_name', models.CharField(max_length=100)),
                ('pet_type', models.CharField(max_length=100)),
                ('pet_level', models.PositiveIntegerField(default=1)),
                ('pet_level_progress', models.PositiveIntegerField(default=0)),
                ('pet_coin', models.PositiveIntegerField(default=0)),
                ('pet_rank', models.PositiveIntegerField(default=0)),
                ('student', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='pet', to='login_register.student')),
            ],
        ),
        migrations.DeleteModel(
            name='VirtualPet',
        ),
    ]