from django.urls import path
from . import views

app_name ='chat'
urlpatterns = [
    path('guard_officer_chat/', views.guard_officer_chat, name='guard_officer_chat'),
    
    path('guard_chat/', views.guard_chat, name='guard_chat'),
    path('get_guard_room/', views.get_guard_room, name='get_guard_room'),
    
    
    path('client_chat/', views.client_chat, name='client_chat'),
    # path('messages/<int:recipient_id>/', views.messages, name='messages'),
    path('', views.home_content, name='home_content'),    
    
    path('rooms/', views.get_rooms, name='get_rooms'),
    
    path('get_guard_officer/', views.get_guard_officer, name='get_guard_officer'),
    path('get_client_room/', views.get_client_room, name='get_client_room'),
    
    
    
    
    path('messages/<int:room_id>/', views.handle_messages, name='handle_messages'),
    
    path('chatbotmessages/', views.chatbot, name='chabot_message'),
    
    path('guard_officer_chat/<int:room_id>/', views.render_chats, name='render_chats'),

    

    
    
    
    
    # FOR Guard
    path('guard_chat/', views.guard_chat, name='guard_chat'),
    
    
    
    
]
    # path('chatbot/',views.chatbot ,name='chatbot')