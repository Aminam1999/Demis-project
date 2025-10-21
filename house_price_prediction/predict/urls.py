from django.urls import path
from .views import HousePricePrediction

urlpatterns = [
    path('predict/', HousePricePrediction.as_view(), name='predict_price'),
]
