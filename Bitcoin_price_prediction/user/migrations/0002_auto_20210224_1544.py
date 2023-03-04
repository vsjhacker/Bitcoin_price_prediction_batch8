

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='bitcoin_price1',
            new_name='bitcoin_price2',
        ),
    ]
