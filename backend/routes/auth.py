from flask import Blueprint, request, jsonify, session, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
import re

auth_bp = Blueprint('auth', __name__)

def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def is_strong_password(password):
    return len(password) >= 8

def serialize_user(user):
    return {
        'id': str(user['_id']),
        'name': user['name'],
        'email': user['email'],
        'role': user['role'],
        'department': user.get('department')
    }

@auth_bp.route('/register', methods=['POST'])
def register():
    db = current_app.db
    data = request.get_json()

    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    role = data.get('role', 'citizen')

    if not name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400

    if not is_valid_email(email):
        return jsonify({'error': 'Invalid email format'}), 400

    if not is_strong_password(password):
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400

    if role not in ['citizen']:  # Only citizens can self-register
        role = 'citizen'

    if db.users.find_one({'email': email}):
        return jsonify({'error': 'Email already registered'}), 409

    user = {
        'name': name,
        'email': email,
        'password': generate_password_hash(password),
        'role': role,
        'department': None
    }
    result = db.users.insert_one(user)
    user['_id'] = result.inserted_id

    session['user_id'] = str(result.inserted_id)
    session['role'] = role

    return jsonify({'message': 'Registration successful', 'user': serialize_user(user)}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    db = current_app.db
    data = request.get_json()

    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    user = db.users.find_one({'email': email})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['user_id'] = str(user['_id'])
    session['role'] = user['role']

    return jsonify({'message': 'Login successful', 'user': serialize_user(user)}), 200

@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@auth_bp.route('/me', methods=['GET'])
def me():
    db = current_app.db
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'user': serialize_user(user)}), 200