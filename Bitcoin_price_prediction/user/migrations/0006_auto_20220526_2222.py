

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0005_auto_20220526_2218'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='close',
            field=models.CharField(max_length=300),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='date1',
            field=models.CharField(max_length=300),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='high',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='low',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='open',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='volume',
            field=models.CharField(max_length=300),
        ),
    ]
