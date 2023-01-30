from django.urls import path

from api.generate import views

urlpatterns = [path("", views.smilesInput.as_view())]
