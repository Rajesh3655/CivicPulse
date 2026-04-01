# Civic System

A comprehensive civic issue reporting and management system built with Flask (backend) and vanilla HTML/CSS/JavaScript (frontend).

## Features

- **User Authentication**: Register, login, logout functionality
- **Role-based Access**: Citizen, Department Admin, Super Admin roles
- **Issue Reporting**: Citizens can report civic issues with location
- **Duplicate Detection**: Automatic detection of duplicate issues using Haversine distance
- **Department Management**: CRUD operations for departments
- **Analytics Dashboard**: Charts and statistics for administrators
- **Responsive UI**: Dark theme with animations

## Project Structure

```
civic-system/
├── start.sh                    # Run this to start the application
├── README.md                   # This documentation
├── backend/
│   ├── app.py                  # Flask app + MongoDB + seed data
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Configuration (SECRET_KEY, MONGO_URI)
│   └── routes/
│       ├── auth.py             # Authentication endpoints
│       ├── issues.py           # Issue CRUD + duplicate detection
│       ├── admin.py            # Analytics + user management
│       └── departments.py      # Department CRUD
└── frontend/
    ├── index.html              # Login/Register page
    ├── citizen.html            # Citizen dashboard
    ├── deptadmin.html          # Department admin dashboard
    └── superadmin.html         # Super admin dashboard
```

## Prerequisites

- Python 3.8+
- MongoDB (local or cloud instance)
- Git

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd civic-system
   ```

2. Install dependencies:

   ```bash
   ./start.sh
   ```

   This will create a virtual environment, install dependencies, and start the server.

3. Or manually:

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Configure environment variables in `backend/.env`:

   ```
   SECRET_KEY=your-secret-key-here
   MONGO_URI=mongodb://localhost:27017/civic_system
   ```

5. Start the application:
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run
   ```

## API Endpoints

### Authentication

- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `POST /auth/logout` - Logout user
- `GET /auth/me` - Get current user info

### Issues

- `GET /issues` - Get all issues (filtered by role)
- `POST /issues` - Create new issue
- `PUT /issues/<id>` - Update issue
- `DELETE /issues/<id>` - Delete issue

### Departments

- `GET /departments` - Get all departments
- `POST /departments` - Create department
- `PUT /departments/<id>` - Update department
- `DELETE /departments/<id>` - Delete department

### Admin

- `GET /admin/analytics` - Get analytics data
- `GET /admin/users` - Get all users
- `PUT /admin/users/<id>` - Update user role
- `DELETE /admin/users/<id>` - Delete user

## Database Schema

### Users

```json
{
  "_id": ObjectId,
  "username": String,
  "email": String,
  "password": String (hashed),
  "role": "citizen" | "dept_admin" | "super_admin",
  "department_id": ObjectId (for dept_admin),
  "created_at": DateTime
}
```

### Issues

```json
{
  "_id": ObjectId,
  "title": String,
  "description": String,
  "location": {
    "lat": Float,
    "lng": Float,
    "address": String
  },
  "status": "pending" | "in_progress" | "resolved" | "closed",
  "department_id": ObjectId,
  "user_id": ObjectId,
  "created_at": DateTime,
  "updated_at": DateTime
}
```

### Departments

```json
{
  "_id": ObjectId,
  "name": String,
  "description": String,
  "created_at": DateTime
}
```

## Technologies Used

- **Backend**: Flask, PyMongo, Flask-JWT-Extended, Flask-CORS
- **Database**: MongoDB
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Chart.js
- **Authentication**: JWT tokens
- **Styling**: Custom CSS with dark theme and animations

## Development

### Running Tests

```bash
cd backend
python -m pytest
```

### Training The AI Models

```bash
cd backend
python train_ml.py
```

This trains the category and priority classifiers from issue data in MongoDB, preferring any admin-corrected labels saved through the dashboard.

### Code Formatting

```bash
cd backend
black .
flake8 .
```

### Database Seeding

The application includes seed data that populates the database with sample users, departments, and issues on startup.

## Deployment

1. Set up MongoDB instance
2. Configure production environment variables
3. Use a WSGI server like Gunicorn:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```
4. Set up reverse proxy with Nginx

## Deploy On Render

This project is ready to deploy to Render as a single Python web service. The Flask backend now serves the frontend HTML pages too, which keeps login/session cookies on the same origin and avoids cross-domain auth issues.

### Files added for Render

- `render.yaml` - Render Blueprint config
- `.python-version` - pins Python 3.13

### Environment variables required

- `MONGO_URI` - your MongoDB Atlas connection string or other MongoDB URI
- `SECRET_KEY` - generated automatically if you create the service from `render.yaml`

### Option A: Deploy with Blueprint

1. Push this project to GitHub.
2. In Render, click **New > Blueprint**.
3. Connect the repository that contains this `render.yaml`.
4. When prompted, provide a value for `MONGO_URI`.
5. Wait for the deploy to finish.

Render will use:

- Build command: `pip install -r backend/requirements.txt`
- Start command: `gunicorn --chdir backend app:app`
- Health check: `/health`

### Option B: Deploy manually as a Web Service

If you do not want to use the Blueprint:

1. In Render, click **New > Web Service**.
2. Connect your repo.
3. Set the root directory to the project root if needed.
4. Use:
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `gunicorn --chdir backend app:app`
5. Add environment variables:
   - `MONGO_URI`
   - `SECRET_KEY`
6. After deploy, open your Render URL. The app landing page is `/` and the API health check is `/health`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
