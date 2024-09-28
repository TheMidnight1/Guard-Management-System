from channels.middleware import BaseMiddleware
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from urllib.parse import parse_qs
from channels.db import database_sync_to_async
import logging

User = get_user_model()

class QueryAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        query_string = scope['query_string']
        query_params = parse_qs(query_string.decode())
        token = query_params.get('token', [None])[0]
        logging.info(f"Token received: {token}")
        scope['user'] = await self.get_user(token)
        logging.info(f"User from token: {scope['user']}")
        return await super().__call__(scope, receive, send)

    @database_sync_to_async
    def get_user(self, token):
        if not token:
            logging.warning("No token provided")
            return AnonymousUser()
        try:
            user = User.objects.get(auth_token=token)  # Ensure this matches your token field
            logging.info(f"User found: {user}")
            return user
        except User.DoesNotExist:
            logging.warning("User not found for token")
            return AnonymousUser()
