

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0004_auto_20220526_2200'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bitcoin_price3',
            name='volume',
            field=models.FloatField(),
        ),
    ]
