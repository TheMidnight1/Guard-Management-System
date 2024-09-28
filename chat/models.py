from django.db import models
from django.conf import settings

# Create your models here.


class ChatRoom(models.Model):
    participants = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='chat_rooms')

    def __str__(self):
        return f'{self.id}'

class Message(models.Model):
    read = models.BooleanField(default=False)
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='sent_messages')    
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE, related_name='messages')
    
    
    def __str__(self):
        return f'{self.id}'
    
    def to_dict(self):
        return {
            'id': self.id,
            'read': self.read,
            'sender_id': self.sender.id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'room_id': self.room.id,
        }
    

