from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt, get_jwt_identity
from pymongo import MongoClient
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from flask_cors import CORS
import torch
from datetime import timedelta, datetime, timezone
from dotenv import load_dotenv
import os


load_dotenv()

model_path = './Roberta_women_safety_final'
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)  
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

client = MongoClient(os.getenv('MONGODB_URI'))
db = client['auth_db']
users_collection = db['users']
data_collection = db['reviews']  

blacklist = set()

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload['jti']
    return jti in blacklist

def predict_anomaly(dialogue, model, tokenizer):
    inputs = tokenizer(dialogue, return_tensors='pt', padding=True, truncation=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    label = 'normal' if predicted_class == 0 else 'anomalous'
    return label

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dialogue = data.get('dialogue', '')
    if not dialogue:
        return jsonify({'error': 'No dialogue provided'}), 400
    
    if '*' in dialogue : 
        return jsonify({'dialogue': dialogue, 'predicted_label': "anomalous"})

    result = predict_anomaly(dialogue, model, tokenizer)
    return jsonify({'dialogue': dialogue, 'predicted_label': result})

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if not all([name, email, password, confirm_password]):
        return jsonify({"message": "Please fill in all fields"}), 400

    if password != confirm_password:
        return jsonify({"message": "Passwords do not match"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "Email already registered"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user = {
        "name": name,
        "email": email,
        "password": hashed_password,
    }
    users_collection.insert_one(user)
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({"message": "Please fill in all fields"}), 400

    user = users_collection.find_one({"email": email})
    if not user or not bcrypt.check_password_hash(user['password'], password):
        return jsonify({"message": "Invalid credentials"}), 401

    access_token = create_access_token(identity={"email": user['email']})
    return jsonify({
        "token": access_token,
        "user": {
            "name": user['name'],
            "email": user['email'],
        }
    }), 200


@app.route('/api/bar-data', methods=['GET'])
def get_bar_data():
    data = data_collection.aggregate([
        {"$group": {"_id": "$place", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ])
    
    labels = []
    values = []
    for item in data:
        labels.append(item['_id'])
        values.append(item['count'])

    return jsonify({"labels": labels, "values": values})

@app.route('/api/pie-data', methods=['GET'])
def get_pie_data():
    data = data_collection.aggregate([
        {"$group": {"_id": "$place", "count": {"$sum": 1}}}
    ])
    
    labels = []
    values = []
    for item in data:
        labels.append(item['_id'])
        values.append(item['count'])

    return jsonify({"labels": labels, "values": values})

@app.route('/api/reviews', methods=['POST'])
def submit_review():
    data = request.get_json()
    name = data.get('name')
    place = data.get('place')
    description = data.get('description')

    if not all([name, place, description]):
        return jsonify({"message": "Please fill in all fields"}), 400

    review = {
        "name": name,
        "place": place,
        "description": description
    }
    data_collection.insert_one(review)
    return jsonify({"message": "Review submitted successfully"}), 201

# Jwt verification
@app.route('/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    return jsonify({"message": "Token is valid"}), 200


@app.route('/api/user', methods=['GET'])
@jwt_required()
def get_user_details():
    current_user = get_jwt_identity()
    
    user = users_collection.find_one({"email": current_user['email']})
    if not user:
        return jsonify({"message": "User not found"}), 404
    
    return jsonify({
        "name": user['name'],
        "email": user['email']
    }), 200

@app.route('/api/reviews', methods=['GET'])
def get_reviews():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    skip = (page - 1) * per_page
    
    reviews = list(data_collection.find().skip(skip).limit(per_page))
    reviews = [{
        'id': str(review['_id']),
        'name': review.get('name'),
        'place': review.get('place'),
        'description': review.get('description')
    } for review in reviews]
    
    return jsonify(reviews)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT'))
