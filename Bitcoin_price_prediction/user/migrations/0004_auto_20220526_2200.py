

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0003_auto_20210224_1614'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='close',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='date1',
            field=models.DateField(),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='high',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='low',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='open',
            field=models.FloatField(),
        ),
    ]
