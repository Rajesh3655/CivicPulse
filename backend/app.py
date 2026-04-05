from flask import Flask, abort, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import os
import tempfile
from dotenv import load_dotenv
from train_ml import ensure_model_ready_async
from cloudinary_service import configure_cloudinary, is_cloudinary_enabled

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key:
    raise ValueError("SECRET_KEY environment variable is required")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
SEEDED_UPLOADS_DIR = os.path.join(BASE_DIR, 'seed_uploads')
FRONTEND_PAGES = {
    'index.html',
    'citizen.html',
    'deptadmin.html',
    'superadmin.html'
}

# Session cookie security
# Browsers reject SameSite=None cookies unless Secure=True.
# For local HTTP development, keep the cookie same-site so session auth works.
is_development = os.environ.get('FLASK_ENV') == 'development'
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=not is_development,
    SESSION_COOKIE_SAMESITE='Lax' if is_development else 'None'
)

# Store uploads outside the project workspace so local static dev servers
# do not auto-refresh the frontend whenever preview/final images are written.
default_upload_folder = os.path.join(tempfile.gettempdir(), 'civic_issue_uploads')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', default_upload_folder)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# CORS configuration - allow local dev origins (including static file servers)
allowed_origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:15500",
    "http://127.0.0.1:15500",
    "http://localhost:5500",
    "http://127.0.0.1:5500"
]
# Allow override from env for staging/dev
if os.environ.get('CORS_ORIGINS'):
    allowed_origins += [o.strip() for o in os.environ.get('CORS_ORIGINS').split(',') if o.strip()]

CORS(app, supports_credentials=True, origins=allowed_origins)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(SEEDED_UPLOADS_DIR, exist_ok=True)

# MongoDB connection
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['civic_issues_db']

def ensure_default_super_admin():
    email = 'admin@civic.gov'
    password_hash = generate_password_hash('admin123')
    existing_user = db.users.find_one({'email': email})

    if existing_user:
        db.users.update_one(
            {'_id': existing_user['_id']},
            {'$set': {
                'name': 'Super Admin',
                'email': email,
                'password': password_hash,
                'role': 'super_admin',
                'department': None
            }}
        )
    else:
        db.users.insert_one({
            'name': 'Super Admin',
            'email': email,
            'password': password_hash,
            'role': 'super_admin',
            'department': None
        })

ensure_default_super_admin()
ensure_model_ready_async(db)
configure_cloudinary()

# Make db accessible to routes
app.db = db

# Register blueprints
from routes.auth import auth_bp
from routes.issues import issues_bp
from routes.admin import admin_bp
from routes.departments import departments_bp

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(issues_bp, url_prefix='/api/issues')
app.register_blueprint(admin_bp, url_prefix='/api/admin')
app.register_blueprint(departments_bp, url_prefix='/api/departments')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(upload_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    seeded_path = os.path.join(SEEDED_UPLOADS_DIR, filename)
    if os.path.exists(seeded_path):
        return send_from_directory(SEEDED_UPLOADS_DIR, filename)

    abort(404)

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_frontend(filename):
    if filename in FRONTEND_PAGES:
        return send_from_directory(FRONTEND_DIR, filename)
    abort(404)

@app.route('/health')
def health():
    return {
        'status': 'ok',
        'message': 'Civic Issue System API Running',
        'cloudinary_enabled': is_cloudinary_enabled()
    }

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return {'error': 'Uploaded image is too large. Please upload a file smaller than 16MB.'}, 413

if __name__ == '__main__':
    debug_mode = is_development
    print("\nSmart Civic Issue Reporting System")
    print("=====================================")
    print(f"API running at: http://localhost:5000 (debug: {debug_mode})")
    print("\nDefault super admin: admin@civic.gov / admin123\n")
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
