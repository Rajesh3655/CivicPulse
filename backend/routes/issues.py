from flask import Blueprint, request, jsonify, session, current_app
from bson import ObjectId
from datetime import datetime, timezone
import os
import math
from werkzeug.utils import secure_filename
from ml_service import analyze_issue

issues_bp = Blueprint('issues', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
DUPLICATE_DISTANCE_THRESHOLD_METERS = 200

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def is_duplicate_distance(distance_meters):
    return distance_meters is not None and distance_meters <= DUPLICATE_DISTANCE_THRESHOLD_METERS

def serialize_utc_timestamp(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat().replace('+00:00', 'Z')

    if isinstance(value, str):
        if value.endswith('Z') or '+' in value[10:] or (value[10:].count('-') > 0):
            return value
        return f'{value}Z'

    return str(value or '')

def serialize_issue(issue):
    return {
        'id': str(issue['_id']),
        'title': issue['title'],
        'description': issue['description'],
        'image': issue.get('image'),
        'resolution_image': issue.get('resolution_image'),
        'latitude': issue.get('latitude'),
        'longitude': issue.get('longitude'),
        'department': issue.get('department'),
        'department_name': issue.get('department_name', 'Unassigned'),
        'status': issue.get('status', 'Pending'),
        'created_at': serialize_utc_timestamp(issue.get('created_at')),
        'updated_at': serialize_utc_timestamp(issue.get('updated_at')),
        'resolution_notes': issue.get('resolution_notes', ''),
        'user_id': str(issue.get('user_id', '')),
        'user_name': issue.get('user_name', 'Unknown'),
        'category': issue.get('category', ''),
        'corrected_category': issue.get('corrected_category', ''),
        'priority': issue.get('priority', 'Medium'),
        'corrected_priority': issue.get('corrected_priority', ''),
        'ai_summary': issue.get('ai_summary', ''),
        'ai_category': issue.get('ai_category', ''),
        'ai_category_confidence': issue.get('ai_category_confidence', 0),
        'ai_department': issue.get('ai_department'),
        'ai_department_name': issue.get('ai_department_name', ''),
        'ai_priority': issue.get('ai_priority', ''),
        'ai_priority_confidence': issue.get('ai_priority_confidence', 0),
        'image_signal': issue.get('image_signal'),
        'image_insight': issue.get('image_insight', ''),
        'image_category': issue.get('image_category'),
        'image_category_confidence': issue.get('image_category_confidence', 0),
        'image_brightness': issue.get('image_brightness'),
        'image_contrast': issue.get('image_contrast'),
        'duplicate_score': issue.get('duplicate_score', 0),
        'duplicate_issue_id': str(issue.get('duplicate_issue_id')) if issue.get('duplicate_issue_id') else None,
        'duplicate_similarity': issue.get('duplicate_similarity', 0),
        'duplicate_distance_meters': issue.get('duplicate_distance_meters'),
        'duplicate_message': issue.get('duplicate_message', ''),
        'model_source': issue.get('model_source', 'bootstrap'),
        'ai_feedback_count': len(issue.get('ai_feedback', [])),
        'timeline': [
            {
                **entry,
                'timestamp': serialize_utc_timestamp(entry.get('timestamp'))
            }
            for entry in issue.get('timeline', [])
        ]
    }

def get_department_for_category(db, category):
    """Find department matching the given category."""
    dept = db.departments.find_one({'categories': {'$in': [category]}})
    if dept:
        return str(dept['_id']), dept['name']
    # Default to Public Safety
    dept = db.departments.find_one({'name': 'Public Safety'})
    if dept:
        return str(dept['_id']), dept['name']
    return None, 'Unassigned'

@issues_bp.route('/', methods=['GET'])
def get_issues():
    db = current_app.db
    user_id = session.get('user_id')
    role = session.get('role')

    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    query = {}
    if role == 'citizen':
        query['user_id'] = ObjectId(user_id)
    elif role == 'dept_admin':
        user = db.users.find_one({'_id': ObjectId(user_id)})
        if user and user.get('department'):
            query['department'] = user['department']

    status_filter = request.args.get('status')
    if status_filter and status_filter != 'all':
        query['status'] = status_filter

    issues = list(db.issues.find(query).sort('created_at', -1))
    return jsonify({'issues': [serialize_issue(i) for i in issues]}), 200

@issues_bp.route('/<issue_id>', methods=['GET'])
def get_issue(issue_id):
    db = current_app.db
    user_id = session.get('user_id')
    role = session.get('role')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        issue = db.issues.find_one({'_id': ObjectId(issue_id)})
        if not issue:
            return jsonify({'error': 'Issue not found'}), 404

        # Authorization: citizens see own issues; dept admins see own department; super admins see all
        if role == 'citizen' and str(issue.get('user_id')) != user_id:
            return jsonify({'error': 'Unauthorized'}), 403

        if role == 'dept_admin':
            user = db.users.find_one({'_id': ObjectId(user_id)})
            if not user or user.get('department') != issue.get('department'):
                return jsonify({'error': 'Unauthorized'}), 403

        return jsonify({'issue': serialize_issue(issue)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@issues_bp.route('/ai-preview', methods=['POST'])
def ai_preview():
    db = current_app.db
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    is_multipart = bool(request.content_type and 'multipart/form-data' in request.content_type)
    data = request.form if is_multipart else (request.get_json() or {})
    title = data.get('title', '').strip()
    description = data.get('description', '').strip()
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    try:
        lat = float(latitude) if latitude not in (None, '') else None
        lon = float(longitude) if longitude not in (None, '') else None
    except (TypeError, ValueError):
        lat, lon = None, None

    if len(title) < 3 and len(description) < 12:
        return jsonify({'error': 'Provide a bit more detail for AI preview'}), 400

    preview_image_path = None
    if is_multipart and 'image' in request.files:
        file = request.files['image']
        if file and file.filename and allowed_file(file.filename):
            preview_filename = secure_filename(f"preview_{datetime.now().timestamp()}_{file.filename}")
            preview_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], preview_filename)
            file.save(preview_image_path)

    try:
        ai_result = analyze_issue(db, title, description, lat, lon, image_path=preview_image_path)
    finally:
        if preview_image_path and os.path.exists(preview_image_path):
            try:
                os.remove(preview_image_path)
            except OSError:
                pass

    return jsonify({
        'preview': {
            'category': ai_result.get('category'),
            'category_confidence': ai_result.get('category_confidence'),
            'department_id': ai_result.get('department_id'),
            'department_name': ai_result.get('department_name'),
            'priority': ai_result.get('priority'),
            'priority_confidence': ai_result.get('priority_confidence'),
            'summary': ai_result.get('summary'),
            'duplicate_score': ai_result.get('duplicate_score'),
            'duplicate_issue_id': ai_result.get('duplicate_issue_id'),
            'duplicate_message': ai_result.get('duplicate_message'),
            'image_insight': ai_result.get('image_insight'),
            'image_category': ai_result.get('image_category'),
            'image_category_confidence': ai_result.get('image_category_confidence'),
            'rule_based_match': ai_result.get('rule_based_match'),
        }
    }), 200

@issues_bp.route('/', methods=['POST'])
def create_issue():
    db = current_app.db
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    category = request.form.get('category', 'Other').strip()
    selected_department = request.form.get('department', '').strip()
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    if not title or not description:
        return jsonify({'error': 'Title and description are required'}), 400

    try:
        lat = float(latitude) if latitude else None
        lon = float(longitude) if longitude else None
    except (ValueError, TypeError):
        lat, lon = None, None

    # Duplicate detection
    if lat is not None and lon is not None:
        existing_issues = db.issues.find({
            'title': title,  # Exact match instead of regex to prevent injection
            'latitude': {'$ne': None},
            'longitude': {'$ne': None}
        })
        for existing in existing_issues:
            existing_lat = existing.get('latitude')
            existing_lon = existing.get('longitude')
            if existing_lat is not None and existing_lon is not None:
                dist = haversine_distance(lat, lon, existing_lat, existing_lon)
                if is_duplicate_distance(dist):
                    return jsonify({
                        'error': 'duplicate',
                        'message': (
                            f'Similar issue already reported within '
                            f'{DUPLICATE_DISTANCE_THRESHOLD_METERS}m ({int(dist)}m away)'
                        ),
                        'existing_id': str(existing['_id'])
                    }), 409

    # Handle image upload
    image_filename = None
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            image_filename = filename

    # Get user info
    user = db.users.find_one({'_id': ObjectId(user_id)})
    user_name = user['name'] if user else 'Unknown'

    image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image_filename) if image_filename else None
    ai_result = analyze_issue(db, title, description, lat, lon, image_path=image_path)

    duplicate_distance = ai_result.get('duplicate_distance_meters')
    if (
        ai_result.get('duplicate_issue_id')
        and is_duplicate_distance(duplicate_distance)
    ):
        return jsonify({
            'error': 'duplicate',
            'message': ai_result.get('duplicate_message') or 'Potential duplicate complaint already exists.',
            'existing_id': ai_result.get('duplicate_issue_id'),
            'duplicate_score': ai_result.get('duplicate_score'),
        }), 409

    # Assign department selected by citizen, with category fallback.
    dept_id, dept_name = None, 'Unassigned'
    if selected_department:
        try:
            dept = db.departments.find_one({'_id': ObjectId(selected_department)})
            if dept:
                dept_id, dept_name = str(dept['_id']), dept['name']
        except Exception:
            dept_id, dept_name = None, 'Unassigned'

    if not dept_id:
        dept_id = ai_result.get('department_id')
        dept_name = ai_result.get('department_name', 'Unassigned')

    if not dept_id:
        dept_id, dept_name = get_department_for_category(db, ai_result.get('category') or category)

    final_category = ai_result.get('category') if category in ['', 'Other'] else category
    final_category = final_category or 'Other'

    now = datetime.now(timezone.utc)
    issue = {
        'title': title,
        'description': description,
        'category': final_category,
        'image': image_filename,
        'latitude': lat,
        'longitude': lon,
        'department': dept_id,
        'department_name': dept_name,
        'status': 'Pending',
        'priority': ai_result.get('priority', 'Medium'),
        'ai_summary': ai_result.get('summary', ''),
        'ai_category': ai_result.get('category', ''),
        'ai_category_confidence': ai_result.get('category_confidence', 0),
        'ai_department': ai_result.get('department_id'),
        'ai_department_name': ai_result.get('department_name', ''),
        'ai_priority': ai_result.get('priority', ''),
        'ai_priority_confidence': ai_result.get('priority_confidence', 0),
        'image_signal': ai_result.get('image_signal'),
        'image_insight': ai_result.get('image_insight', ''),
        'image_category': ai_result.get('image_category'),
        'image_category_confidence': ai_result.get('image_category_confidence', 0),
        'image_brightness': ai_result.get('image_brightness'),
        'image_contrast': ai_result.get('image_contrast'),
        'model_source': ai_result.get('model_source', 'bootstrap'),
        'duplicate_score': ai_result.get('duplicate_score', 0),
        'duplicate_issue_id': ObjectId(ai_result['duplicate_issue_id']) if ai_result.get('duplicate_issue_id') else None,
        'duplicate_similarity': ai_result.get('duplicate_similarity', 0),
        'duplicate_distance_meters': ai_result.get('duplicate_distance_meters'),
        'duplicate_message': ai_result.get('duplicate_message', ''),
        'ai_feedback': [],
        'created_at': now,
        'updated_at': now,
        'resolution_notes': '',
        'resolution_image': None,
        'user_id': ObjectId(user_id),
        'user_name': user_name,
        'timeline': [
            {
                'status': 'Pending',
                'note': (
                    f"Issue submitted by citizen. AI classified this as {final_category} "
                    f"with {ai_result.get('priority', 'Medium')} priority."
                    + (f" Image insight: {ai_result.get('image_insight')}" if ai_result.get('image_insight') else '')
                ),
                'timestamp': now.isoformat()
            }
        ]
    }

    result = db.issues.insert_one(issue)
    issue['_id'] = result.inserted_id

    return jsonify({'message': 'Issue submitted successfully', 'issue': serialize_issue(issue)}), 201

@issues_bp.route('/<issue_id>/status', methods=['PUT'])
def update_status(issue_id):
    db = current_app.db
    user_id = session.get('user_id')
    role = session.get('role')

    if not user_id or role not in ['dept_admin', 'super_admin']:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.form if request.content_type and 'multipart' in request.content_type else request.get_json()
    if not data:
        data = request.get_json() or {}

    new_status = data.get('status')
    resolution_notes = data.get('resolution_notes', '')

    valid_statuses = ['Pending', 'In Progress', 'Resolved']
    if new_status not in valid_statuses:
        return jsonify({'error': f'Invalid status. Must be one of: {valid_statuses}'}), 400

    # Handle resolution image upload
    resolution_image = None
    if hasattr(request, 'files') and 'resolution_image' in request.files:
        file = request.files['resolution_image']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(f"res_{datetime.now().timestamp()}_{file.filename}")
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            resolution_image = filename

    try:
        issue = db.issues.find_one({'_id': ObjectId(issue_id)})
        if not issue:
            return jsonify({'error': 'Issue not found'}), 404

        # Dept admins can only update issues within their department
        if role == 'dept_admin':
            user = db.users.find_one({'_id': ObjectId(user_id)})
            if not user or user.get('department') != issue.get('department'):
                return jsonify({'error': 'Unauthorized to update this issue'}), 403

        now = datetime.now(timezone.utc)
        timeline = issue.get('timeline', [])
        timeline.append({
            'status': new_status,
            'note': resolution_notes or f'Status updated to {new_status}',
            'timestamp': now.isoformat()
        })

        update_data = {
            'status': new_status,
            'resolution_notes': resolution_notes,
            'updated_at': now,
            'timeline': timeline
        }
        if resolution_image:
            update_data['resolution_image'] = resolution_image

        db.issues.update_one({'_id': ObjectId(issue_id)}, {'$set': update_data})

        updated = db.issues.find_one({'_id': ObjectId(issue_id)})
        return jsonify({'message': 'Status updated', 'issue': serialize_issue(updated)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@issues_bp.route('/<issue_id>', methods=['DELETE'])
def delete_issue(issue_id):
    db = current_app.db
    role = session.get('role')
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        issue = db.issues.find_one({'_id': ObjectId(issue_id)})
        if not issue:
            return jsonify({'error': 'Issue not found'}), 404

        if role not in ['super_admin'] and str(issue.get('user_id')) != user_id:
            return jsonify({'error': 'Unauthorized'}), 403

        db.issues.delete_one({'_id': ObjectId(issue_id)})
        return jsonify({'message': 'Issue deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
