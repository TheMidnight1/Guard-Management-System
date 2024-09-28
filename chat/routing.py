from . import event_consumers
from django.urls import re_path

from django.urls import path

websocket_urlpatterns = [
    path(r'ws/', event_consumers.EventConsumer.as_asgi()),

]

