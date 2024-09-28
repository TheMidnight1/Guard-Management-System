import json
from uuid import uuid4
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import StopConsumer

clients = []

class EventConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        try:
            await self.accept()

            # Use the built-in channel_name for unique client ID
            self._id = self.channel_name
            self._user_id = self.scope['user'].id
            
            clients.append(self)
            print(f"Connected new client with channel_name {self._id} and user_id {self._user_id}")
        except Exception as e:
            print(f"Unexpected error during connection: {e}")

    @property
    def id(self):
        return self._id

    @property
    def user_id(self):
        return self._user_id

    async def disconnect(self, close_code):
        global clients
        print(f"Disconnected client with channel_name {self._id}")
        
        # Remove the client from the list
        clients = [client for client in clients if client.id != self._id]

        # Clean up any resources
        raise StopConsumer()

    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            
            # Check for 'event' key in the JSON data
            if "event" in text_data_json:
                event_type = text_data_json["event"]
                match event_type:
                    case "ping":
                        await self.send(text_data=json.dumps({"event": "pong"}))
                    case "send_message":
                        await self.broadcast_message(text_data_json.get("payload", {}))
                    case _:
                        print(f"Unknown event type: {event_type}")
        except Exception as e:
            print(f"Error in receive: {e}")

    async def broadcast_message(self, payload):
        """
        Broadcast the message to all connected clients.
        """
        for client in clients:
            try:
                await client.send(text_data=json.dumps({
                    'event': 'new_message',
                    'payload': payload
                }))
            except Exception as e:
                print(f"Error broadcasting message to client {client.id}: {e}")

    @staticmethod
    async def broadcast(event, payload, user_ids=None):
        """
        Static method to broadcast a message to specific users or all users.
        """
        event_name = event.split(':', 1)[-1]

        if user_ids is not None:
            # Filter clients based on user IDs
            audiences = [client for client in clients if client.user_id and client.user_id in user_ids]
        else:
            audiences = clients

        for audience in audiences:
            try:
                await audience.send(json.dumps({
                    'event': event_name,
                    'payload': payload
                }))
            except Exception as e:
                print(f"Error broadcasting to audience {audience.id}: {e}")
