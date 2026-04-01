from flask import Blueprint, request, jsonify, session, current_app
from bson import ObjectId
import re

departments_bp = Blueprint('departments', __name__)

@departments_bp.route('/', methods=['GET'])
def get_departments():
    db = current_app.db
    departments = list(db.departments.find())
    result = []
    for d in departments:
        result.append({
            'id': str(d['_id']),
            'name': d['name'],
            'categories': d.get('categories', [])
        })
    return jsonify({'departments': result}), 200

@departments_bp.route('/', methods=['POST'])
def create_department():
    db = current_app.db
    if session.get('role') != 'super_admin':
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    name = data.get('name', '').strip()
    categories = data.get('categories', [])

    if not name:
        return jsonify({'error': 'Department name required'}), 400

    if not isinstance(categories, list):
        categories = [str(categories)]

    if db.departments.find_one({'name': {'$regex': f'^{re.escape(name)}$', '$options': 'i'}}):
        return jsonify({'error': 'Department already exists'}), 409

    if db.departments.find_one({'name': name}):
        return jsonify({'error': 'Department already exists'}), 409

    dept = {'name': name, 'categories': categories}
    result = db.departments.insert_one(dept)
    return jsonify({'message': 'Department created', 'id': str(result.inserted_id)}), 201

@departments_bp.route('/<dept_id>', methods=['PUT'])
def update_department(dept_id):
    db = current_app.db
    if session.get('role') != 'super_admin':
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    update = {}
    if 'name' in data: update['name'] = data['name']
    if 'categories' in data: update['categories'] = data['categories']

    try:
        db.departments.update_one({'_id': ObjectId(dept_id)}, {'$set': update})
        return jsonify({'message': 'Department updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@departments_bp.route('/<dept_id>', methods=['DELETE'])
def delete_department(dept_id):
    db = current_app.db
    if session.get('role') != 'super_admin':
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        # Prevent deleting a department with assigned users or issues
        user_count = db.users.count_documents({'department': dept_id})
        issue_count = db.issues.count_documents({'department': dept_id})
        if user_count > 0 or issue_count > 0:
            return jsonify({'error': 'Cannot delete department with assigned users or issues'}), 400

        db.departments.delete_one({'_id': ObjectId(dept_id)})
        return jsonify({'message': 'Department deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400