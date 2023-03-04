from django.db import models

# Create your models here.
class user_reg(models.Model):
    id = models.AutoField(primary_key=True)
    fullname = models.CharField(max_length=300)
    email = models.CharField(max_length=200)
    mobile = models.CharField(max_length=200)
    uname = models.CharField(max_length=200)
    password = models.CharField(max_length=300)

class bitcoin_price3(models.Model):
    id = models.AutoField(primary_key=True)
    date1 = models.CharField(max_length=300)
    open = models.CharField(max_length=200)
    high = models.CharField(max_length=200)
    low = models.CharField(max_length=200)
    close = models.CharField(max_length=300)
    volume = models.CharField(max_length=300)