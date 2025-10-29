# Tripster Backend Configuration Guide

## Required Environment Variables

Create a `.env` file in the Backend directory with the following variables:

```
# Flask Configuration
SECRET_KEY=your-secret-key-change-this-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-this-in-production

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/tripster

# Google Cloud Configuration
GOOGLE_PROJECT_ID=your-google-project-id
GOOGLE_LOCATION=us-central1

# API Keys
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
OPENWEATHERMAP_API_KEY=your-openweathermap-api-key
```

## How to Get API Keys

### Google Maps API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the following APIs:
   - Maps JavaScript API
   - Places API
   - Geocoding API
   - Air Quality API
4. Create credentials (API Key)
5. Restrict the API key to your domain/IP for security

### OpenWeatherMap API Key
1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Get your API key from the dashboard

### MongoDB Setup
1. Install MongoDB locally: https://www.mongodb.com/try/download/community
2. Or use MongoDB Atlas (cloud): https://www.mongodb.com/atlas

### Google Cloud Project Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable Vertex AI API
4. Set up Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```

## Running the Application

1. Install dependencies:
   ```bash
   cd Backend
   pip install -r requirements.txt
   ```

2. Set up environment variables (create .env file)

3. Run the backend:
   ```bash
   python run.py
   ```

4. Run the frontend (in another terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
