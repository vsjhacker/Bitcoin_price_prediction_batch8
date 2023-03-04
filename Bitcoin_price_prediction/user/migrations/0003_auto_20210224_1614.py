

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0002_auto_20210224_1544'),
    ]

    operations = [
        migrations.CreateModel(
            name='bitcoin_price3',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('date1', models.CharField(max_length=300)),
                ('open', models.CharField(max_length=200)),
                ('high', models.CharField(max_length=200)),
                ('low', models.CharField(max_length=200)),
                ('close', models.CharField(max_length=300)),
                ('volume', models.CharField(max_length=300)),
            ],
        ),
        migrations.DeleteModel(
            name='bitcoin_price2',
        ),
    ]
