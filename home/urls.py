from . import views
from django.urls import path

app_name = 'home'
urlpatterns = [
    path('', views.home, name='home'),
    path('img/', views.home_img, name='img'),
    path('results_img/', views.home_results_img, name='home_results_img'),
    path('video/', views.home_video, name='video'),
    path('results_video/', views.home_results_video, name='home_results_video'),
    path('webcam/', views.webcam, name='webcam'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('download_video/<str:path>', views.download_video, name='download_video'),
]
