from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
			path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),
			path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
			path("UploadDataset", views.UploadDataset, name="UploadDataset"),
			path("UploadDatasetAction", views.UploadDatasetAction, name="UploadDatasetAction"),
			path("ProcessDataset", views.ProcessDataset, name="ProcessDataset"),
			path("TrainVGG", views.TrainVGG, name="TrainVGG"),
			path("TrainCNN", views.TrainCNN, name="TrainCNN"),
			path("Predict", views.Predict, name="Predict"),
			path("PredictAction", views.PredictAction, name="PredictAction"),			
]