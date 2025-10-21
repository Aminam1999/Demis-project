from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import xgboost as xgb
from django.http import JsonResponse
import joblib

# بارگذاری مدل XGBoost از فایل با استفاده از joblib
model = joblib.load('xgboost_model.joblib')


class HousePricePrediction(APIView):
    def post(self, request):
        try:
            # دریافت ویژگی‌ها از درخواست JSON
            features = np.array(request.data['features']).reshape(1, -1)

            # پیش‌بینی قیمت
            prediction = model.predict(features)

            return JsonResponse({'predicted_price': prediction[0]}, status=status.HTTP_200_OK)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
