from flask import Blueprint, request, jsonify
from . import mongo, bcrypt # Import shared extensions
from pymongo.errors import DuplicateKeyError
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager # Added JWT functions
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from bson import ObjectId # Needed to convert user ID string back to MongoDB ObjectId
from .services import generate_itinerary_ai # Import our AI service function

# --- Authentication Blueprint ---
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    """Registers a new user."""
    data = request.get_json()

    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({"success": False, "message": "Missing username, email, or password"}), 400

    username = data['username']
    email = data['email']
    password = data['password']

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    try:
        user_id = mongo.db.users.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password
        }).inserted_id
        print(f"User created with ID: {user_id}")
        return jsonify({"success": True, "message": "User registered successfully!"}), 201

    except DuplicateKeyError as e:
        error_field = "Unknown field"
        if 'username' in e.details.get('keyPattern', {}):
             error_field = "Username"
        elif 'email' in e.details.get('keyPattern', {}):
             error_field = "Email"
        return jsonify({"success": False, "message": f"{error_field} is already taken!"}), 400

    except Exception as e:
        print(f"Error during signup: {e}")
        return jsonify({"success": False, "message": "An error occurred during registration."}), 500

@auth_bp.route('/signin', methods=['POST'])
def signin():
    """Authenticates a user and returns a JWT."""
    data = request.get_json()

    if not data or not data.get('username') or not data.get('password'):
        return jsonify({"success": False, "message": "Missing username or password"}), 400

    username = data['username']
    password = data['password']

    user = mongo.db.users.find_one({'username': username})

    if user and bcrypt.check_password_hash(user['password'], password):
        user_id_str = str(user['_id'])
        access_token = create_access_token(identity=user_id_str)
        return jsonify(accessToken=access_token), 200
    else:
        return jsonify({"success": False, "message": "Invalid username or password"}), 401

# --- Itinerary Blueprint ---
itinerary_bp = Blueprint('itinerary', __name__)

@itinerary_bp.route('/generate', methods=['POST'])
@jwt_required() # Protect this route
def generate_plan():
    """Generates a travel itinerary using AI, requires JWT authentication."""
    # Get user ID from the JWT token
    current_user_id = get_jwt_identity()
    user = mongo.db.users.find_one({'_id': ObjectId(current_user_id)})

    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    # Get itinerary request details from JSON body
    request_data = request.get_json()
    if not request_data or not request_data.get('destination') or not request_data.get('numberOfDays') or request_data.get('budget') is None:
         return jsonify({"success": False, "message": "Missing destination, numberOfDays, or budget"}), 400

    # Call the AI service function from services.py
    result = generate_itinerary_ai(request_data, user['username'])

    # Handle potential errors from the AI service
    if "error" in result:
        status_code = 400 if result.get("code") == "BUDGET_TOO_LOW" else 500
        return jsonify({"success": False, "message": result["error"], "code": result.get("code")}), status_code
    else:
        # Return the successful AI response
        return jsonify(result), 200