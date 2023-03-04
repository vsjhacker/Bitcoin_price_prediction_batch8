

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='bitcoin_price1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date1', models.CharField(max_length=300)),
                ('open', models.CharField(max_length=200)),
                ('high', models.CharField(max_length=200)),
                ('low', models.CharField(max_length=200)),
                ('close', models.CharField(max_length=300)),
                ('volume', models.CharField(max_length=300)),
            ],
        ),
        migrations.CreateModel(
            name='user_reg',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('fullname', models.CharField(max_length=300)),
                ('email', models.CharField(max_length=200)),
                ('mobile', models.CharField(max_length=200)),
                ('uname', models.CharField(max_length=200)),
                ('password', models.CharField(max_length=300)),
            ],
        ),
    ]
