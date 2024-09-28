"""
ASGI config for smartpatrol project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/asgi/
"""

# import os

# from channels.routing import ProtocolTypeRouter
# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartpatrol.settings')

# application = ProtocolTypeRouter({
#     "http": get_asgi_application()
# })
import os
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from chat.middleware import QueryAuthMiddleware
import chat.routing  

from chat.routing import websocket_urlpatterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartpatrol.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
        "websocket":AllowedHostsOriginValidator(
        AuthMiddlewareStack(
                
            URLRouter(
                chat.routing.websocket_urlpatterns
            )
        )
    )
})
