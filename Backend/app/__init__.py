import os
from flask import Flask, jsonify
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Initialize extensions (but don't connect them to the app yet)
mongo = PyMongo()
bcrypt = Bcrypt()
jwt = JWTManager()
cors = CORS()

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)

    # --- Configuration ---
    # Load secret key for JWT and Flask session management from .env
    # Provide fallbacks just in case, though .env should always be used
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "default-flask-secret-key-change-me")
    app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "default-jwt-secret-key-change-me")

    # Load MongoDB connection string from .env (fallback to local dev)
    mongo_uri = os.getenv("MONGO_URI") or "mongodb://localhost:27017/tripster"
    if not os.getenv("MONGO_URI"):
        print("WARNING: MONGO_URI not set. Using default local MongoDB at mongodb://localhost:27017/tripster")
    app.config["MONGO_URI"] = mongo_uri

    # You can add other configurations here if needed, e.g., JWT expiration time
    from datetime import timedelta
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24) # Tokens expire in 24 hours

    # --- Initialize Extensions with App ---
    # Connect the initialized extensions to the Flask app instance
    mongo.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)
    # CORS: Allow localhost for dev, and Netlify domains for production
    # In production, allow all origins (you can restrict to specific domain via FRONTEND_URL)
    is_production = os.getenv("RENDER") is not None or os.getenv("ENVIRONMENT") == "production"
    if is_production:
        # Production: Allow all origins (or set specific via FRONTEND_URL)
        frontend_url = os.getenv("FRONTEND_URL")
        if frontend_url:
            cors.init_app(app, origins=[frontend_url], supports_credentials=True)
        else:
            # Allow all origins in production if FRONTEND_URL not set
            cors.init_app(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
    else:
        # Development: Only localhost
        cors.init_app(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"], supports_credentials=True)

    # --- JWT Error Handlers ---
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({"success": False, "message": "Token has expired. Please log in again."}), 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({"success": False, "message": "Invalid token. Please log in again."}), 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({"success": False, "message": "Authorization token is required. Please log in."}), 401

    # --- Register Blueprints (API Routes) ---
    # Import the blueprints defined in routes.py
    from .routes import auth_bp, itinerary_bp

    # Register the blueprints with the app and specify URL prefixes
    app.register_blueprint(auth_bp, url_prefix='/api/auth') # Auth routes will be under /api/auth/...
    app.register_blueprint(itinerary_bp, url_prefix='/api/itinerary') # Itinerary routes under /api/itinerary/...

    # --- Application Context ---
    # Use the application context to perform actions that require the app to be set up,
    # like interacting with the database extensions.
    with app.app_context():
        # Import the index creation function from models.py
        from .models import create_user_indexes
        # Attempt to create database indexes on startup
        create_user_indexes()
        print("Flask App created, extensions initialized, and DB indexes checked/created.")

    # Return the configured Flask app instance
    return app