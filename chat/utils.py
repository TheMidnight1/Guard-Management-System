import asyncio
import websockets
import json

async def send_message(room_name, message):
    async with websockets.connect('ws://localhost:6789') as websocket:
        await websocket.send(room_name)
        await websocket.send(json.dumps({'message': message, 'room': room_name}))

def emit_event(room_name, message):
    asyncio.get_event_loop().run_until_complete(send_message(room_name, message))
