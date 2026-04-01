from flask import Blueprint, request, jsonify, session, current_app
from bson import ObjectId
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash
import re
from ml_service import load_trained_artifacts
from train_ml import trigger_training_async

admin_bp = Blueprint('admin', __name__)


def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def is_strong_password(password):
    return len(password) >= 8

def require_admin(roles=['super_admin', 'dept_admin']):
    def decorator(f):
        def wrapper(*args, **kwargs):
            user_id = session.get('user_id')
            role = session.get('role')
            if not user_id:
                return jsonify({'error': 'Not authenticated'}), 401
            if role not in roles:
                return jsonify({'error': 'Unauthorized'}), 403
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator

@admin_bp.route('/analytics', methods=['GET'])
def analytics():
    db = current_app.db
    user_id = session.get('user_id')
    role = session.get('role')
    if not user_id or role not in ['super_admin', 'dept_admin']:
        return jsonify({'error': 'Unauthorized'}), 403

    # Build filter based on role
    match_filter = {}
    if role == 'dept_admin':
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if user and user.get('department'):
            match_filter['department'] = user['department']

    total = db.issues.count_documents(match_filter)
    pending = db.issues.count_documents({**match_filter, 'status': 'Pending'})
    in_progress = db.issues.count_documents({**match_filter, 'status': 'In Progress'})
    resolved = db.issues.count_documents({**match_filter, 'status': 'Resolved'})
    high_priority = db.issues.count_documents({**match_filter, 'priority': 'High'})
    medium_priority = db.issues.count_documents({**match_filter, 'priority': 'Medium'})
    low_priority = db.issues.count_documents({**match_filter, 'priority': 'Low'})
    duplicate_candidates = db.issues.count_documents({**match_filter, 'duplicate_score': {'$gte': 0.45}})

    ai_category_pipeline = [
        {'$match': {**match_filter, 'ai_category': {'$nin': [None, '']}}},
        {'$group': {'_id': '$ai_category', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}},
        {'$limit': 6}
    ]
    ai_categories = list(db.issues.aggregate(ai_category_pipeline))

    priority_pipeline = [
        {'$match': {**match_filter, 'priority': {'$in': ['High', 'Medium', 'Low']}}},
        {'$group': {'_id': '$priority', 'count': {'$sum': 1}}}
    ]
    priority_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for item in db.issues.aggregate(priority_pipeline):
        priority_counts[item['_id']] = item['count']

    # Issues per department
    pipeline = [
        {'$match': match_filter} if match_filter else {'$match': {}},
        {'$group': {'_id': '$department_name', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}}
    ]
    by_dept = list(db.issues.aggregate(pipeline))

    # Issues over time (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    time_pipeline = [
        {'$match': {**match_filter, 'created_at': {'$gte': thirty_days_ago}}},
        {'$group': {
            '_id': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}},
            'count': {'$sum': 1}
        }},
        {'$sort': {'_id': 1}}
    ]
    over_time = list(db.issues.aggregate(time_pipeline))
    model_info = None
    if role == 'super_admin':
        trained = load_trained_artifacts()
        metadata = trained.metadata or {}
        model_info = {
            'source': 'trained' if trained.category_model is not None else 'bootstrap',
            'trained_at': metadata.get('trained_at'),
            'dataset_size': metadata.get('dataset_size', 0),
            'category_accuracy': metadata.get('category_model', {}).get('accuracy'),
            'priority_accuracy': metadata.get('priority_model', {}).get('accuracy'),
            'category_labels': metadata.get('category_labels', {}),
            'priority_labels': metadata.get('priority_labels', {}),
            'retrain_command': 'cd backend && python train_ml.py',
            'notes': [
                metadata.get('category_model', {}).get('note'),
                metadata.get('priority_model', {}).get('note'),
            ],
        }

    # Fill missing dates
    date_map = {d['_id']: d['count'] for d in over_time}
    dates = []
    counts = []
    for i in range(30):
        date = (thirty_days_ago + timedelta(days=i+1)).strftime('%Y-%m-%d')
        dates.append(date)
        counts.append(date_map.get(date, 0))

    return jsonify({
        'summary': {
            'total': total,
            'pending': pending,
            'in_progress': in_progress,
            'resolved': resolved
        },
        'ai_summary': {
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority,
            'duplicate_candidates': duplicate_candidates,
            'duplicate_rate': round((duplicate_candidates / total) * 100, 1) if total else 0,
        },
        'by_department': [{'name': d['_id'] or 'Unassigned', 'count': d['count']} for d in by_dept],
        'by_ai_category': [{'name': d['_id'] or 'Uncategorized', 'count': d['count']} for d in ai_categories],
        'over_time': {'dates': dates, 'counts': counts},
        'status_distribution': {
            'Pending': pending,
            'In Progress': in_progress,
            'Resolved': resolved
        },
        'priority_distribution': priority_counts,
        'ai_highlights': [
            f'{high_priority} high-priority issues currently need attention.',
            f'{duplicate_candidates} complaints were flagged as likely duplicates.',
            (
                f"Top predicted issue type: {(ai_categories[0]['_id'] if ai_categories else 'No AI categories yet')}"
            )
        ],
        'model_info': model_info,
    }), 200

@admin_bp.route('/users', methods=['GET'])
def get_users():
    db = current_app.db
    if session.get('role') != 'super_admin':
        return jsonify({'error': 'Unauthorized'}), 403

    users = list(db.users.find({}, {'password': 0}))
    result = []
    for u in users:
        dept_name = None
        if u.get('department'):
            try:
                dept = db.departments.find_one({'_id': ObjectId(u['department'])})
                dept_name = dept['name'] if dept else None
            except:
                pass
        result.append({
            'id': str(u['_id']),
            'name': u['name'],
            'email': u['email'],
            'role': u['role'],
            'department': u.get('department'),
            'department_name': dept_name
        })
    return jsonify({'users': result}), 200

@admin_bp.route('/users', methods=['POST'])
def create_user():
    db = current_app.db
    if session.get('role') != 'super_admin':
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    role = data.get('role', 'citizen')
    department = data.get('department')

    if not name or not email or not password:
        return jsonify({'error': 'All fields required'}), 400

    if not is_valid_email(email):
        return jsonify({'error': 'Invalid email format'}), 400

    if not is_strong_password(password):
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400

    if db.users.find_one({'email': email}):
        return jsonify({'error': 'Email already exists'}), 409

    if role not in ['citizen', 'dept_admin', 'super_admin']:
        role = 'citizen'

    if role not in ['citizen', 'dept_admin', 'super_admin']:
        role = 'citizen'

    if role == 'dept_admin' and not department:
        return jsonify({'error': 'Department is required for Department Admin'}), 400

    if role != 'dept_admin':
        department = None

    user = {
        'name': name,
        'email': email,
        'password': generate_password_hash(password),
        'role': role,
        'department': department
    }
    result = db.users.insert_one(user)
    return jsonify({'message': 'User created', 'id': str(result.inserted_id)}), 201

@admin_bp.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    db = current_app.db
    if session.get('role') != 'super_admin':
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    update = {}
    if 'name' in data: update['name'] = data['name']
    if 'role' in data and data['role'] in ['citizen', 'dept_admin', 'super_admin']:
        update['role'] = data['role']
    if 'department' in data: update['department'] = data['department']

    if update.get('role') == 'dept_admin' and not update.get('department'):
        return jsonify({'error': 'Department is required for Department Admin'}), 400
    if update.get('role') in ['citizen', 'super_admin']:
        update['department'] = None

    if 'password' in data and data['password']:
        if not is_strong_password(data['password']):
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        update['password'] = generate_password_hash(data['password'])

    try:
        db.users.update_one({'_id': ObjectId(user_id)}, {'$set': update})
        return jsonify({'message': 'User updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@admin_bp.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    db = current_app.db
    if session.get('role') != 'super_admin':
        return jsonify({'error': 'Unauthorized'}), 403

    if user_id == session.get('user_id'):
        return jsonify({'error': 'Cannot delete yourself'}), 400

    try:
        db.users.delete_one({'_id': ObjectId(user_id)})
        return jsonify({'message': 'User deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@admin_bp.route('/issues/<issue_id>/ai-feedback', methods=['PUT'])
def update_ai_feedback(issue_id):
    db = current_app.db
    user_id = session.get('user_id')
    role = session.get('role')

    if not user_id or role not in ['super_admin', 'dept_admin']:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json() or {}
    corrected_category = (data.get('corrected_category') or '').strip()
    corrected_priority = (data.get('corrected_priority') or '').strip()
    notes = (data.get('notes') or '').strip()

    if not corrected_category and not corrected_priority and not notes:
        return jsonify({'error': 'No feedback provided'}), 400

    try:
        issue = db.issues.find_one({'_id': ObjectId(issue_id)})
        if not issue:
            return jsonify({'error': 'Issue not found'}), 404

        if role == 'dept_admin':
            user = db.users.find_one({'_id': ObjectId(user_id)})
            if not user or user.get('department') != issue.get('department'):
                return jsonify({'error': 'Unauthorized'}), 403

        feedback_entry = {
            'reviewed_by': user_id,
            'reviewed_role': role,
            'corrected_category': corrected_category or issue.get('category'),
            'corrected_priority': corrected_priority or issue.get('priority'),
            'notes': notes,
            'created_at': datetime.utcnow(),
        }

        update_data = {
            'updated_at': datetime.utcnow(),
            'ai_feedback': issue.get('ai_feedback', []) + [feedback_entry],
        }
        if corrected_category:
            update_data['corrected_category'] = corrected_category
        if corrected_priority:
            update_data['corrected_priority'] = corrected_priority

        db.issues.update_one({'_id': ObjectId(issue_id)}, {'$set': update_data})
        trigger_training_async(db)
        return jsonify({'message': 'AI feedback saved'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@admin_bp.route('/all-issues', methods=['GET'])
def get_all_issues():
    db = current_app.db
    role = session.get('role')
    user_id = session.get('user_id')

    if not user_id or role not in ['super_admin', 'dept_admin']:
        return jsonify({'error': 'Unauthorized'}), 403

    query = {}
    if role == 'dept_admin':
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if user and user.get('department'):
            query['department'] = user['department']

    status_filter = request.args.get('status')
    if status_filter and status_filter != 'all':
        query['status'] = status_filter

    from routes.issues import serialize_issue
    issues = list(db.issues.find(query).sort('created_at', -1))
    return jsonify({'issues': [serialize_issue(i) for i in issues]}), 200
