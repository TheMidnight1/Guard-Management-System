import json
from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth import get_user_model
from .models import Message
from asgiref.sync import sync_to_async
from django.db.models import Q

User = get_user_model()

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.room_name

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json.get('message')
        fetch_history = text_data_json.get('fetch_history')
        sender = self.scope["user"]

        if fetch_history:
            await self.fetch_message_history(sender)
        elif message:
            await self.handle_new_message(sender, message)

    async def fetch_message_history(self, sender):
        room_parts = self.room_name.split('_')
        if len(room_parts) != 3:
            await self.send(text_data=json.dumps({'error': f'Invalid room name format: {self.room_name}'}))
            return

        sender_id = room_parts[1]
        recipient_id = room_parts[2]

        sender = await sync_to_async(User.objects.get)(id=sender_id)
        recipient = await sync_to_async(User.objects.get)(id=recipient_id)

        messages = await sync_to_async(list)(
            Message.objects.filter(
                (Q(sender=sender) & Q(recipient=recipient)) |
                (Q(sender=recipient) & Q(recipient=sender))
            ).order_by('timestamp').values('sender__email', 'content', 'timestamp')
        )

        for message in messages:
            message['timestamp'] = message['timestamp'].isoformat()

        await self.send(text_data=json.dumps({'messages': messages}))

    async def handle_new_message(self, sender, message):
        room_parts = self.room_name.split('_')
        if len(room_parts) != 3:
            await self.send(text_data=json.dumps({'error': f'Invalid room name format: {self.room_name}'}))
            return

        sender_id = room_parts[1]
        recipient_id = room_parts[2]

        if str(sender.id) != sender_id:
            await self.send(text_data=json.dumps({'error': 'Sender ID mismatch.'}))
            return

        recipient = await sync_to_async(User.objects.get)(id=recipient_id)
        if recipient is None:
            await self.send(text_data=json.dumps({'error': f'User with ID {recipient_id} does not exist.'}))
            return

        msg = await sync_to_async(Message.objects.create)(sender=sender, recipient=recipient, content=message)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'sender_email': sender.email,
                'timestamp': msg.timestamp.isoformat()
            }
        )

    async def chat_message(self, event):
        message = event['message']
        sender_email = event['sender_email']
        timestamp = event['timestamp']

        await self.send(text_data=json.dumps({
            'message': message,
            'sender_email': sender_email,
            'timestamp': timestamp
        }))
