import websockets
import os
import json
import joblib
import asyncio
from .models import Message,ChatRoom
from . import event_consumers
from django.conf import settings
from main_app.models import Client,Guard,GuardOfficeUser
from django.db.models import Q

from django.db.models import OuterRef, Subquery
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render,get_object_or_404
from django.contrib.auth.decorators import login_required
import os
import json
import joblib
import asyncio
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity



# Download NLTK resources if not already present
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the model and vectorizer from saved files
MODEL_PATH = os.path.join(settings.BASE_DIR, 'chat', 'naive_bayes_model_nice.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'chat', 'tfidf_vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Path to the CSV file containing the chatbot data
DATA_PATH = os.path.join(settings.BASE_DIR, 'chat', 'smart_petrol_expanded.csv')

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Utility function to clean text
def clean_text(text):
    """Preprocess text by lowercasing, removing special characters, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Clean both questions and answers in the dataset
data['Cleaned_Question'] = data['Question'].apply(clean_text)
data['Cleaned_Answer'] = data['Answer'].apply(clean_text)


# Function to generate a response from the chatbot model
def generate_response(question):
    # Clean and vectorize the input question
    cleaned_question = clean_text(question)
    input_vector = vectorizer.transform([cleaned_question])
    
    # Predict using the Naive Bayes model
    predicted_index = model.predict(input_vector)[0]
    answer = data['Answer'].iloc[predicted_index]
    
    return answer

# Chatbot view to handle POST requests
def chatbot(request):
    if request.method == 'POST':
        # Parse the incoming JSON data
        data = json.loads(request.body)
        message = data['message']
        
        # Generate response using the model
        response = generate_response(message)
        print(response)
        # Broadcast the response to WebSocket clients (if using WebSocket)
        asyncio.run(event_consumers.EventConsumer.broadcast(
            "private:new-message",
            {"message": {"content": response}},
            [request.user.id]
        ))

        # Return the response to the client
        return JsonResponse({'response': response})
    
    # Render the chatbot HTML page for GET requests
    return render(request, 'client_template/chatbot.html')

def retrieve_similar_question(user_input, top_k=3):
    # Vectorize all questions for cosine similarity
    all_questions_vector = vectorizer.transform(data['Cleaned_Question'])
    input_vector = vectorizer.transform([user_input])
    
    # Calculate cosine similarity between input and all questions
    similarities = cosine_similarity(input_vector, all_questions_vector).flatten()
    
    # Get the top-k similar questions based on cosine similarity
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    
    # Use the best match index and its score
    best_match_idx = top_k_indices[0]
    best_score = similarities[best_match_idx]
    
    return best_match_idx, best_score

def jaccard_similarity(user_input):
    # Tokenize user input
    cleaned_input = clean_text(user_input)
    input_set = set(cleaned_input.split())

    # Initialize best score and index
    best_score = 0
    best_match_idx = None
    
    # Calculate Jaccard similarity with all questions
    for idx, question in enumerate(data['Cleaned_Question']):
        question_set = set(question.split())
        intersection = len(input_set.intersection(question_set))
        union = len(input_set.union(question_set))
        
        # Ensure no division by zero
        if union == 0:
            continue
        
        score = intersection / union
        if score > best_score:
            best_score = score
            best_match_idx = idx
            
    return best_match_idx, best_score


def generate_response(question, nb_threshold=0.7, similarity_threshold=0.2, jaccard_threshold=0.3):
    try:
        # Clean and vectorize the input question
        cleaned_question = clean_text(question)
        input_vector = vectorizer.transform([cleaned_question])
        
        # Step 1: Naive Bayes prediction
        predicted_index = model.predict(input_vector)[0]
        confidence_score = max(model.predict_proba(input_vector)[0])
        
        if confidence_score >= nb_threshold:
            return data['Answer'].iloc[predicted_index]
        
        # Step 2: Use cosine similarity as a fallback
        best_match_idx, similarity_score = retrieve_similar_question(cleaned_question)
        if similarity_score >= similarity_threshold:
            return data['Answer'].iloc[best_match_idx]
        
        # Step 3: Use Jaccard similarity as a last fallback
        best_match_idx, jaccard_score = jaccard_similarity(cleaned_question)
        if jaccard_score >= jaccard_threshold:
            return data['Answer'].iloc[best_match_idx]
        
    except Exception as e:
        print(f"Error generating response: {e}")
    
    # If no confident match is found or an error occurs
    return "I'm not sure about that. Please contact the admin for more details."


def get_room_messages(user, room_id,limit=30):
    
        chat_room, _ = ChatRoom.objects.get_or_create(id=room_id)
        chat_room.participants.add(user)
        messages = Message.objects.filter(room=chat_room).order_by('timestamp')[:limit]
        messages_list = [
            {
                "read": message.read,
                "content": message.content,
                "sent": message.sender.id==user.id,
                "timestamp": message.timestamp.isoformat()
                
            }
            for message in messages
        ]
        
        
        return messages_list
    
    
def get_user_rooms(user):
    # Get all chat rooms where the request.user is a participant
    chat_rooms = ChatRoom.objects.filter(participants=user)

    # Subquery to get the latest message for each chat room
    latest_message_subquery = Message.objects.filter(room=OuterRef('pk')).order_by('-timestamp').values('pk')[:1]

    # Annotate each chat room with the latest message
    chat_rooms = chat_rooms.annotate(latest_message_id=Subquery(latest_message_subquery))

    # Prepare response data
    rooms = []
    for room in chat_rooms:
        latest_message = Message.objects.filter(pk=room.latest_message_id).first()

        participants = room.participants.all()

        rooms.append({
            "id": room.id,
            "latest_message": {
                "read": latest_message.read if latest_message else None,
                "sender": latest_message.sender.id if latest_message else None,
                "content": latest_message.content if latest_message else None,
                "timestamp": latest_message.timestamp if latest_message else None
            } if latest_message else None,
            "participants": [
                {
                    "id": participant.id,
                    "email": participant.email,
                    "profile_pic": participant.profile_pic
                    
                } for participant in participants
            ]
        })

    return rooms


@login_required
def guard_officer_chat(request):
    rooms = get_user_rooms(request.user)
    return render(request, 'guardofficeuser_template/officer_chat.html', {'rooms': rooms})


@login_required
def guard_chat(request):
    # rooms = get_user_rooms(request.user)
    return render(request, 'guard_template/guard_chat.html')

@login_required
def client_chat(request):
    client = request.user.client
    guard_officer = client.guard_office
    return render(request, 'client_template/client_chat.html', {'guard_officer': guard_officer})

@login_required
def home_content(request):
    return render(request, 'chat/home_content.html')

@login_required
def get_rooms(request):
    rooms = get_user_rooms(request.user)
    return JsonResponse({ 'rooms': rooms })



@login_required
def handle_messages(request, room_id):
    
    if request.method=="GET":
        try:
            limit = int(request.GET.get('limit'))
        except (TypeError, ValueError):
            limit = 30
        
        messages_list = get_room_messages(request.user,room_id,limit)
       
        return JsonResponse({ 'messages':messages_list })
    
    elif request.method == 'POST':

        data = json.loads(request.body)
        room, _ = ChatRoom.objects.get_or_create(id=room_id)
        message = Message.objects.create(room=room,content=data['message'], sender=request.user)

        participants = room.participants.exclude(id=request.user.id)
        participants_list = [participant.id for participant in participants]

        asyncio.run(event_consumers.EventConsumer.broadcast("private:new-message", {
            'message': message.to_dict()
        }, participants_list))
        

        
        return JsonResponse({})
    else:
        return HttpResponse(f"Method {request.method} not allowed", status=405)


# def chatbot(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         message = data['message']
#         to = data['message']


#         response = generate_response(message)
#         asyncio.run(event_consumers.EventConsumer.broadcast(
#             "private:new-message",
#             { "message": {'content':response} },
#             [request.user.id])
#         )

#         return JsonResponse({'response': response })
    
#     return render(request, 'client_template/chatbot.html')


# def generate_response(question):
#     predicted_index = model.predict([question])[0]
#     answer = encoder.inverse_transform([predicted_index])
#     return answer[0]

@login_required
def get_guard_officer(request):
    client = get_object_or_404(Client, admin=request.user)
    
    return JsonResponse({
        'id':client.guard_office.id
    })
    
@login_required
def get_client_room(request):
    # room, _ = ChatRoom.objects.get_or_create()
    client = get_object_or_404(Client, admin=request.user)
    guard_office_user = GuardOfficeUser.objects.filter(guard_office=client.guard_office).first()
   
    room_exist = False
    user_room = ChatRoom.objects.filter(participants=request.user.id)
    for r in user_room:
        if r.participants.filter(id=guard_office_user.admin.id).exists():
            room_exist=True
            room = r;
    
    if not room_exist:
        room = ChatRoom.objects.create()
        room.participants.set([request.user, guard_office_user.admin])
        room.save()

    return JsonResponse({
        'id':room.id,
    })
    


# For Guard
@login_required
def guard_chat(request):
    if hasattr(request.user, 'guard'):
        guard_office = request.user.guard.guard_office
        guards = Guard.objects.filter(guard_office=guard_office)
        
        return render(request, 'guard_template/guard_chat.html', {'guards': guards})
    else:
        # Handle the case where the user is not a guard
        return render(request, 'guard_template/guard_chat.html', {'guards': []})

@login_required
def render_chats(request,room_id):

    rooms = get_user_rooms(request.user)
    messages = get_room_messages(request.user,room_id)
    
    return render(request, 'guardofficeuser_template/officer_chat.html', {'rooms': rooms,'message_list':messages})



@login_required
def get_guard_room(request):
    guard = get_object_or_404(Guard, admin=request.user)
    guard_office_user = GuardOfficeUser.objects.filter(guard_office=guard.guard_office).first()

    room_exist = False
    user_room = ChatRoom.objects.filter(participants=request.user.id)
    for r in user_room:
        if r.participants.filter(id=guard_office_user.admin.id).exists():
            room_exist = True
            room = r

    if not room_exist:
        room = ChatRoom.objects.create()
        room.participants.set([request.user, guard_office_user.admin])
        room.save()

    return JsonResponse({'id': room.id})

@login_required
def guard_chat(request):
    return render(request, 'guard_template/guard_chat.html')