import os
import vertexai
# Ensure necessary imports from vertexai.generative_models are present
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
from google.auth import default # Import the default credentials finder
from google.auth.exceptions import DefaultCredentialsError # Import specific exception
import locale # For currency formatting
import requests # For making HTTP requests (Weather, Air Quality)
import googlemaps # For Google Maps geocoding
from datetime import datetime, timedelta # For processing weather forecast dates
import json # For parsing potential JSON responses
from urllib.parse import urlencode # For creating URL parameters
import pandas as pd # For data manipulation
from sklearn.cluster import KMeans # For geographical clustering
from sklearn.preprocessing import StandardScaler # For scaling coordinates before clustering
import numpy as np # For numerical operations
import random # For shuffling cluster tops in selection
import math # For distance calculations


# --- Google Maps Client Initialization ---
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
IXIGO_API_KEY = os.getenv("IXIGO_API_KEY")
IRCTC_API_KEY = os.getenv("IRCTC_API_KEY")
MAKEMYTRIP_API_KEY = os.getenv("MAKEMYTRIP_API_KEY")

gmaps = None
if GOOGLE_MAPS_API_KEY:
	try:
		gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
		print("Google Maps client initialized successfully.")
	except Exception as e:
		print(f"ERROR: Failed to initialize Google Maps client: {e}")
		gmaps = None
else:
	print("ERROR: GOOGLE_MAPS_API_KEY not found in .env file. Geocoding, weather, air quality, POI fetching, and smart links will fail.")


# --- Vertex AI Initialization ---
def initialize_vertexai():
	"""Initializes the Vertex AI client using ADC, with explicit checks."""
	try:
		project_id = os.getenv("GOOGLE_PROJECT_ID")
		location = os.getenv("GOOGLE_LOCATION")
		if not project_id or not location:
			print("ERROR: GOOGLE_PROJECT_ID or GOOGLE_LOCATION not set in .env file.")
			return False

		print(f"Attempting to initialize Vertex AI for project: {project_id} in location: {location}")
		try:
			credentials, found_project_id = default()
			adc_project_msg = f"Found Google Cloud credentials via ADC. ADC Quota Project: {found_project_id}"
			print(adc_project_msg)
			if found_project_id != project_id:
				 print(f"WARNING: ADC quota project ('{found_project_id}') does not match configured GOOGLE_PROJECT_ID ('{project_id}'). Ensure ADC quota project has Vertex AI enabled/billing linked, or run 'gcloud auth application-default set-quota-project {project_id}'.")
			vertexai.init(project=project_id, location=location, credentials=credentials)
		except DefaultCredentialsError:
			print("ERROR: Could not find Application Default Credentials (ADC). Run 'gcloud auth application-default login'.")
			return False
		print(f"Vertex AI initialized successfully for project: {project_id}")
		return True
	except Exception as e:
		print(f"ERROR: Unexpected error initializing Vertex AI: {e}")
		return False

VERTEX_AI_INITIALIZED = initialize_vertexai()
# --- Use the user-requested model name ---
MODEL_NAME = "gemini-2.5-pro" # Using the requested Gemini model name


# --- Weather Function ---
def get_weather_forecast(destination, num_days):
	"""Fetches weather forecast using OpenWeatherMap API after geocoding."""
	if not gmaps: return "Weather info unavailable (Maps client missing)."
	openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
	if not openweathermap_api_key: return "Weather info unavailable (Weather API key missing)."
	try:
		print(f"Geocoding '{destination}' for weather...")
		geocode_result = gmaps.geocode(destination)
		if not geocode_result: return f"Weather info unavailable (Cannot find coordinates for {destination})."
		lat = geocode_result[0]['geometry']['location']['lat']; lon = geocode_result[0]['geometry']['location']['lng']
		print(f"Coordinates for weather: Lat={lat}, Lon={lon}")
		forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={openweathermap_api_key}&units=metric"
		print("Fetching weather forecast..."); response = requests.get(forecast_url, timeout=10)
		response.raise_for_status(); weather_data = response.json(); print("Weather forecast data received.")
		forecast_summary = f"Weather Forecast Summary for {destination}:\n"; daily_forecasts = {}; today = datetime.now().date()
		if 'list' in weather_data:
			for forecast in weather_data['list']:
				dt = datetime.fromtimestamp(forecast['dt']); forecast_date = dt.date()
				if forecast_date < today: continue
				date_str = forecast_date.strftime('%Y-%m-%d')
				if date_str not in daily_forecasts:
					if len(daily_forecasts) >= num_days: continue
					daily_forecasts[date_str] = {'min_temp': 1000.0, 'max_temp': -1000.0, 'conditions': set()}
				temp = forecast.get('main', {}).get('temp')
				if temp is not None:
					daily_forecasts[date_str]['min_temp'] = min(daily_forecasts[date_str]['min_temp'], temp)
					daily_forecasts[date_str]['max_temp'] = max(daily_forecasts[date_str]['max_temp'], temp)
				if forecast.get('weather'):
					condition = forecast['weather'][0].get('main');
					if condition: daily_forecasts[date_str]['conditions'].add(condition)
			if not daily_forecasts: forecast_summary += "No specific forecast data available for upcoming days.\n"
			else:
				 for date_str, data in sorted(daily_forecasts.items()):
					 day_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%a, %b %d')
					 min_t_str = f"{data['min_temp']:.0f}°C" if data['min_temp'] != 1000.0 else "N/A"
					 max_t_str = f"{data['max_temp']:.0f}°C" if data['max_temp'] != -1000.0 else "N/A"
					 conditions_str = ', '.join(sorted(list(data['conditions']))) if data['conditions'] else 'Varied'
					 forecast_summary += (f"- {day_date}: Temp {min_t_str} - {max_t_str}, Conditions: {conditions_str}\n")
		else:
			print("Warning: 'list' key not found in OpenWeatherMap response."); forecast_summary += "Could not parse forecast data.\n"
		return forecast_summary.strip()
	# (Keep detailed error handling)
	except googlemaps.exceptions.ApiError as e: print(f"ERROR: Google Maps API Error (Weather Geocoding): {e}"); return "Weather info unavailable (Geocoding error)."
	except requests.exceptions.Timeout: print(f"ERROR: Timeout connecting to OpenWeatherMap."); return "Weather info unavailable (Timeout)."
	except requests.exceptions.RequestException as e: print(f"ERROR: OpenWeatherMap API request failed: {e}"); return "Weather info unavailable (Connection error)." # Simplified
	except Exception as e: print(f"ERROR: Unexpected error processing weather data: {e}"); return "Weather info unavailable (Processing error)."

# --- Air Quality Function ---
def get_air_quality(destination):
	"""Fetches current air quality using Google Air Quality API."""
	if not gmaps: return "Air Quality info unavailable (Maps client missing)."
	if not GOOGLE_MAPS_API_KEY: return "Air Quality info unavailable (Maps API key missing)."
	try:
		print(f"Geocoding '{destination}' for air quality...")
		geocode_result = gmaps.geocode(destination)
		if not geocode_result: return f"Air Quality info unavailable (Cannot find coordinates for {destination})."
		lat = geocode_result[0]['geometry']['location']['lat']; lon = geocode_result[0]['geometry']['location']['lng']
		print(f"Coordinates for Air Quality: Lat={lat}, Lon={lon}")
		aq_url = "https://airquality.googleapis.com/v1/currentConditions:lookup"; params = {'key': GOOGLE_MAPS_API_KEY}
		payload = {"location": {"latitude": lat, "longitude": lon}}; headers = {'Content-Type': 'application/json'}
		print("Fetching air quality data..."); response = requests.post(aq_url, params=params, json=payload, headers=headers, timeout=10)
		response.raise_for_status(); aq_data = response.json(); print("Air quality data received.")
		if aq_data and 'indexes' in aq_data and len(aq_data['indexes']) > 0:
			aqi_info = next((idx for idx in aq_data['indexes'] if idx.get('code') == 'uaqi'), aq_data['indexes'][0])
			aqi_value = aqi_info.get('aqiDisplay', 'N/A'); aqi_category = aqi_info.get('category', 'N/A'); dominant_pollutant = aqi_info.get('dominantPollutant', 'N/A')
			summary = f"AQI: {aqi_value} ({aqi_category}). Dominant Pollutant: {dominant_pollutant}."
			return summary.strip()
		else:
			print("Warning: Unexpected structure in Air Quality API response."); return "Air Quality data currently unavailable."
	# (Keep detailed error handling)
	except googlemaps.exceptions.ApiError as e: print(f"ERROR: Google Maps API Error (AQI Geocoding): {e}"); return "Air Quality info unavailable (Geocoding error)."
	except requests.exceptions.Timeout: print(f"ERROR: Timeout connecting to Google Air Quality API."); return "Air Quality info unavailable (Timeout)."
	except requests.exceptions.RequestException as e: print(f"ERROR: Google Air Quality API request failed: {e}"); return "Air Quality info unavailable (Connection/Permission error)." # Simplified
	except Exception as e: print(f"ERROR: Unexpected error processing air quality data: {e}"); return "Air Quality info unavailable (Processing error)."

# --- Travel Options Function ---

# --- Real API Functions ---
def get_flight_options(origin_city, destination_city, start_date_str):
	"""Fetches real flight options using RapidAPI (Flight Data by Api-Ninja)."""
	if not RAPIDAPI_KEY:
		return {"error": "RAPIDAPI_KEY not configured"}
	
	# City to IATA code mapping
	city_to_iata = {
		"delhi": "DEL", "mumbai": "BOM", "bangalore": "BLR", "goa": "GOI",
		"kolkata": "CCU", "chennai": "MAA", "hyderabad": "HYD", "pune": "PNQ",
		"ahmedabad": "AMD", "kochi": "COK", "trivandrum": "TRV", "mysore": "MYQ"
	}
	
	origin_iata = city_to_iata.get(origin_city.lower())
	dest_iata = city_to_iata.get(destination_city.lower())
	
	if not origin_iata or not dest_iata:
		return {"error": f"Could not find airport codes for {origin_city} to {destination_city}"}

	try:
		date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
		formatted_date = date_obj.strftime('%Y-%m-%d')
	except Exception:
		formatted_date = datetime.now().strftime('%Y-%m-%d')

	api_url = 'https://flight-data-by-api-ninjas.p.rapidapi.com/v1/flights'
	params = {
		'origin': origin_iata,
		'destination': dest_iata,
		'date': formatted_date
	}
	headers = {
		"X-RapidAPI-Key": RAPIDAPI_KEY,
		"X-RapidAPI-Host": "flight-data-by-api-ninjas.p.rapidapi.com"
	}

	try:
		print(f"Fetching flight data for: {origin_iata} to {dest_iata} on {formatted_date}")
		response = requests.get(api_url, headers=headers, params=params, timeout=10)
		response.raise_for_status()
		flights = response.json()
		
		if not flights:
			return {"available": False, "message": "No flights found for this route/date"}

		# Get cheapest flight
		cheapest_flight = min(flights, key=lambda x: x.get('price', float('inf')))
		
		return {
			"available": True,
			"airline": cheapest_flight.get('airline', 'N/A'),
			"flight_number": cheapest_flight.get('flight_number', 'N/A'),
			"price_inr": f"₹{cheapest_flight.get('price', 0):,}",
			"departure_at": cheapest_flight.get('departure_at', 'N/A'),
			"description": f"Cheapest flight via {cheapest_flight.get('airline')}",
			"booking_links": {
				"search_url": f"https://www.ixigo.com/search/result/flight?from={origin_iata}&to={dest_iata}&date={formatted_date}",
				"api_available": True
			}
		}

	except requests.exceptions.RequestException as e:
		print(f"ERROR: RapidAPI (Flights) request failed: {e}")
		return {"available": False, "error": f"Flight API error: {e}"}
	except Exception as e:
		print(f"ERROR: Processing flight data failed: {e}")
		return {"available": False, "error": "Failed to process flight data"}


def get_hotel_options(destination, start_date_str, num_days, adults_number=2, room_number=1):
	"""Fetches real hotel options using RapidAPI (Booking.com)."""
	if not RAPIDAPI_KEY:
		return {"error": "RAPIDAPI_KEY not configured"}
	
	try:
		checkin_date = datetime.strptime(start_date_str, '%Y-%m-%d')
		checkout_date = checkin_date + timedelta(days=num_days)
		checkin_str = checkin_date.strftime('%Y-%m-%d')
		checkout_str = checkout_date.strftime('%Y-%m-%d')
	except Exception:
		checkin_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
		checkout_str = (datetime.now() + timedelta(days=num_days + 1)).strftime('%Y-%m-%d')

	api_url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
	querystring = {
		"dest_type": "city",
		"name": destination,
		"checkin_date": checkin_str,
		"checkout_date": checkout_str,
		"order_by": "popularity",
		"adults_number": str(max(1, int(adults_number or 1))),
		"room_number": str(max(1, int(room_number or 1))),
		"units": "metric",
		"locale": "en-gb",
		"currency": "INR"
	}
	headers = {
		"X-RapidAPI-Key": RAPIDAPI_KEY,
		"X-RapidAPI-Host": "booking-com.p.rapidapi.com"
	}

	try:
		print(f"Fetching hotel data for: {destination} from {checkin_str} to {checkout_str}")
		response = requests.get(api_url, headers=headers, params=querystring, timeout=15)
		response.raise_for_status()
		data = response.json()
		
		hotels = data.get('result', [])
		
		if not hotels:
			return {"available": False, "message": "No hotels found for these dates"}

		# Format top 3 hotels
		hotel_options = []
		for hotel in hotels[:3]:
			name = hotel.get('hotel_name', 'N/A')
			price = hotel.get('price_breakdown', {}).get('gross_price', 'N/A')
			rating = hotel.get('review_score', 'N/A')
			hotel_options.append({
				"name": name,
				"price": f"₹{price}" if isinstance(price, (int, float)) else str(price),
				"rating": rating,
				"description": f"{name} (Rating: {rating}/10, Price: ₹{price})"
			})
		
		return {
			"available": True,
			"hotels": hotel_options,
			"booking_links": {
				"search_url": f"https://www.booking.com/searchresults.html?ss={destination}&checkin={checkin_str}&checkout={checkout_str}",
				"api_available": True
			}
		}

	except requests.exceptions.RequestException as e:
		print(f"ERROR: RapidAPI (Hotels) request failed: {e}")
		return {"available": False, "error": f"Hotel API error: {e}"}
	except Exception as e:
		print(f"ERROR: Processing hotel data failed: {e}")
		return {"available": False, "error": "Failed to process hotel data"}


def get_irctc_links(origin_city, destination_city, date_str=None):
	"""Generate working IRCTC train search links."""
	date_str = date_str or datetime.now().strftime('%Y-%m-%d')
	links = {
		"search_url": f"https://www.irctc.co.in/nget/train-search?from={origin_city}&to={destination_city}&date={date_str}",
		"note": f"Search trains from {origin_city} to {destination_city} on {date_str}"
	}
	return links


def get_bus_links(origin_city, destination_city, date_str=None):
	keyword = f"bus {origin_city} to {destination_city}"
	return generate_smart_links(keyword)


def get_travel_options(destination, origin_city="Delhi"):
	"""Fetches travel options (train/flight) to destination."""
	travel_options = {
		"flight": {
			"available": True,
			"duration": "2-4 hours",
			"price_range": "₹3,000 - ₹8,000",
			"airlines": ["IndiGo", "SpiceJet", "Air India", "Vistara"],
			"description": "Fastest way to reach your destination"
		},
		"train": {
			"available": True,
			"duration": "8-24 hours",
			"price_range": "₹500 - ₹3,000",
			"classes": ["AC 1st", "AC 2nd", "AC 3rd", "Sleeper"],
			"description": "Scenic journey with comfortable amenities"
		},
		"bus": {
			"available": True,
			"duration": "12-36 hours",
			"price_range": "₹300 - ₹1,500",
			"types": ["Volvo", "Semi-sleeper", "Sleeper"],
			"description": "Budget-friendly option with multiple stops"
		}
	}
	
	# Add destination-specific adjustments
	if "Goa" in destination:
		travel_options["flight"]["price_range"] = "₹2,500 - ₹6,000"
		travel_options["train"]["duration"] = "12-18 hours"
	elif "Kerala" in destination:
		travel_options["flight"]["price_range"] = "₹3,500 - ₹7,000"
		travel_options["train"]["duration"] = "16-24 hours"
	elif "Rajasthan" in destination:
		travel_options["flight"]["price_range"] = "₹2,000 - ₹5,000"
		travel_options["train"]["duration"] = "6-12 hours"
	
	# Attach booking/search links
	travel_options["flight"]["booking_links"] = get_ixigo_links(origin_city, destination)
	travel_options["train"]["booking_links"] = get_irctc_links(origin_city, destination)
	travel_options["bus"]["booking_links"] = get_bus_links(origin_city, destination)
	return travel_options

# --- Smart Link Generation Function ---

def generate_smart_links(place_keyword):
	"""Generates Google Maps search and directions links for a place keyword."""
	links = {}
	if not place_keyword: return links
	try:
		encoded_keyword_query = urlencode({"query": place_keyword})
		links['search_url'] = f"https://www.google.com/maps/search/?api=1&{encoded_keyword_query}"
		encoded_keyword_google = urlencode({"q": place_keyword})
		links['google_search_url'] = f"https://www.google.com/search?{encoded_keyword_google}"
	except Exception as e:
		print(f"Warning: Error generating smart links for '{place_keyword}': {e}")
	return links

# --- Fetch Points of Interest Function ---

def fetch_points_of_interest(destination, preferences, dietary_preference=None):
	"""Fetches potential points of interest using Google Places API."""
	if not gmaps: print("Error: Google Maps client not initialized. Cannot fetch POIs."); return [], [], []
	poi_radius_meters = 20000; max_results_per_type = 20
	attraction_keywords = ['tourist attraction', 'landmark', 'park', 'museum', 'historical place']
	restaurant_keywords = ['restaurant', 'cafe', 'food']
	dp = (dietary_preference or '').strip().lower()
	if dp == 'veg' or dp == 'vegetarian':
		restaurant_keywords.extend(['vegetarian restaurant', 'vegan', 'veg thali'])
	elif dp == 'non-veg' or dp == 'nonveg' or dp == 'non vegetarian' or dp == 'non vegetarian':
		restaurant_keywords.extend(['non veg restaurant', 'barbecue', 'meat', 'seafood', 'biryani'])
	elif dp == 'both':
		restaurant_keywords.extend(['vegetarian restaurant', 'vegan', 'veg thali', 'non veg restaurant', 'barbecue', 'meat', 'seafood', 'biryani'])
	if preferences:
		attraction_keywords.extend([p for p in preferences if p.lower() not in ['food', 'restaurant', 'cafe', 'hotel', 'accommodation', 'stay']])
		restaurant_keywords.extend([p for p in preferences if p.lower() in ['food', 'restaurant', 'cafe']])
		attraction_keywords = sorted(list(set(attraction_keywords)))
		restaurant_keywords = sorted(list(set(restaurant_keywords)))
	attractions = []; restaurants = []; hotels = [] # Added hotels here too for consistency if needed later
	try:
		print(f"Geocoding '{destination}' for POI search...")
		geocode_result = gmaps.geocode(destination)
		if not geocode_result: print(f"Warning: Geocoding failed for {destination}."); return [], [], []
		lat = geocode_result[0]['geometry']['location']['lat']; lon = geocode_result[0]['geometry']['location']['lng']
		location_coords = (lat, lon); print(f"Coordinates for POI search: {location_coords}")

		# Search Attractions
		print(f"Searching nearby attractions...")
		nearby_attractions = gmaps.places_nearby(location=location_coords, radius=poi_radius_meters, type='tourist_attraction').get('results', [])
		print(f"Searching attractions with keywords: {attraction_keywords}...")
		text_query_attr = f"{' OR '.join(attraction_keywords)} in {destination}"; text_attractions = gmaps.places(query=text_query_attr, language='en').get('results', [])
		all_attractions_dict = {p['place_id']: p for p in nearby_attractions + text_attractions if p.get('place_id')}
		attractions = list(all_attractions_dict.values())[:max_results_per_type * 2]
		print(f"Found {len(attractions)} unique potential attractions.")

		# Search Restaurants
		print(f"Searching restaurants with keywords: {restaurant_keywords}...")
		text_query_rest = f"{' OR '.join(restaurant_keywords)} in {destination}"; text_restaurants = gmaps.places(query=text_query_rest, language='en').get('results', [])
		all_restaurants_dict = {p['place_id']: p for p in text_restaurants if p.get('place_id')}
		restaurants = list(all_restaurants_dict.values())[:max_results_per_type * 2]
		print(f"Found {len(restaurants)} unique potential restaurants.")

		# Search Hotels
		print("Searching for nearby hotels/lodging...")
		places_result_hotels = gmaps.places_nearby(location=location_coords, radius=poi_radius_meters, type='lodging').get('results', [])
		hotels = list({p['place_id']: p for p in places_result_hotels if p.get('place_id')}.values())[:max_results_per_type] # Deduplicate hotels too
		print(f"Found {len(hotels)} unique potential hotels nearby.")

		# Data Extraction Helper
		def extract_poi_details(place):
			 lat, lng = (None, None); loc = place.get('geometry', {}).get('location', {})
			 if loc: lat, lng = loc.get('lat'), loc.get('lng')
			 photo_ref = place.get('photos', [{}])[0].get('photo_reference') if place.get('photos') else None
			 return {'name': place.get('name'), 'place_id': place.get('place_id'), 'lat': lat, 'lng': lng,
					 'rating': place.get('rating'), 'user_ratings_total': place.get('user_ratings_total'),
					 'types': place.get('types', []), 'address': place.get('formatted_address') or place.get('vicinity'),
					 'photo_reference': photo_ref, 'business_status': place.get('business_status')}

		# Apply extraction and filter
		cleaned_attractions = [extract_poi_details(p) for p in attractions if p.get('name') and p.get('geometry') and p.get('business_status') == 'OPERATIONAL']
		cleaned_restaurants = [extract_poi_details(p) for p in restaurants if p.get('name') and p.get('geometry') and p.get('business_status') == 'OPERATIONAL']
		cleaned_hotels = [extract_poi_details(h) for h in hotels if h.get('name') and h.get('geometry') and h.get('business_status') == 'OPERATIONAL']
		print(f"Filtered to {len(cleaned_attractions)} attractions, {len(cleaned_restaurants)} restaurants, {len(cleaned_hotels)} hotels.")
		return cleaned_attractions, cleaned_restaurants, cleaned_hotels
	except googlemaps.exceptions.ApiError as e: print(f"ERROR: Google Maps API Error (POI Search): {e}"); return [], [], []
	except Exception as e: print(f"ERROR: Unexpected error fetching POIs: {e}"); return [], [], []


# --- ML Helper Functions ---

def preprocess_pois(poi_list):
	"""Converts POI list to DataFrame, cleans data."""
	if not poi_list: return pd.DataFrame()
	df = pd.DataFrame(poi_list); print(f"Preprocessing {len(df)} POIs...")
	df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(2.5)
	df['user_ratings_total'] = pd.to_numeric(df['user_ratings_total'], errors='coerce').fillna(0).astype(int)
	df['lat'] = pd.to_numeric(df['lat'], errors='coerce'); df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
	df['types'] = df['types'].apply(lambda x: x if isinstance(x, list) else [])
	df.dropna(subset=['name', 'lat', 'lng', 'place_id'], inplace=True)
	df = df[(df['rating'] >= 3.5) | (df['user_ratings_total'] < 10)] # Keep higher rated or less rated
	print(f"Preprocessed {len(df)} POIs remaining."); return df.reset_index(drop=True)


def cluster_attractions(attractions_df, num_clusters_hint):
	"""Applies K-Means clustering based on lat/lon. Adds 'cluster' column."""
	if attractions_df.empty or attractions_df[['lat', 'lng']].isnull().values.any():
		print("Warning: Cannot perform clustering - empty/invalid attraction data.");
		if not attractions_df.empty: attractions_df['cluster'] = 0
		return attractions_df
	coordinates = attractions_df[['lat', 'lng']].values; scaler = StandardScaler(); scaled_coordinates = scaler.fit_transform(coordinates)
	k = max(1, min(num_clusters_hint, len(attractions_df) // 2, 5)); print(f"Performing K-Means clustering (k={k})...")
	try:
		kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
		attractions_df['cluster'] = kmeans.fit_predict(scaled_coordinates); print("Clustering complete.")
	except Exception as e: print(f"Error during K-Means: {e}. Assigning default cluster."); attractions_df['cluster'] = 0
	return attractions_df


def rank_pois(pois_df, preferences, preference_weights=None):
    """Ranks POIs using weighted blend of rating, reviews, preference match, and distance-to-centroid.
    Adds 'final_score'. Optional `preference_weights` can override weights.
    """
    if pois_df.empty: return pois_df
    print(f"Ranking {len(pois_df)} POIs based on preferences: {preferences}...")

    # Weights with sensible defaults
    weights = {
        'rating': 0.5,
        'reviews': 0.25,
        'preference': 0.2,
        'distance': 0.05,
    }
    if isinstance(preference_weights, dict):
        for k in weights.keys():
            if k in preference_weights:
                try:
                    weights[k] = float(preference_weights[k])
                except Exception:
                    pass
        # normalize to 1.0
        total_w = sum(weights.values()) or 1.0
        for k in weights:
            weights[k] = weights[k] / total_w

    max_rating = 5.0
    pois_df['rating_score'] = (pois_df['rating'] / max_rating).clip(lower=0, upper=1)

    max_log_reviews = np.log1p(pois_df['user_ratings_total'].max())
    pois_df['review_score'] = (np.log1p(pois_df['user_ratings_total']) / max_log_reviews) if max_log_reviews > 0 else 0.0

    prefs_lower = [p.lower().strip() for p in preferences] if preferences else []

    def preference_score_row(row):
        if not prefs_lower:
            return 0.0
        types = row.get('types', []) or []
        name = (row.get('name') or '').lower()
        types_str_lower = " ".join(types).lower().replace('_', ' ')
        matched = 0
        for pref in prefs_lower:
            if pref in types_str_lower or name.startswith(pref) or name.endswith(pref) or f" {pref} " in f" {types_str_lower} ":
                matched += 1
        return min(1.0, matched / max(1, len(prefs_lower)))

    pois_df['preference_score'] = pois_df.apply(preference_score_row, axis=1)

    # Distance-to-centroid (encourage compact plans). If coords invalid, score = 0.
    try:
        valid_coords = pois_df[['lat','lng']].dropna()
        center_lat = valid_coords['lat'].mean(); center_lng = valid_coords['lng'].mean()
        def distance_score_row(row):
            try:
                lat = float(row['lat']); lng = float(row['lng'])
                if np.isnan(lat) or np.isnan(lng):
                    return 0.0
                # simple haversine-lite using Euclidean on scaled degrees for small areas
                d = np.sqrt((lat - center_lat)**2 + (lng - center_lng)**2)
                # Convert to score in [0,1] where closer is better; 95th percentile as 0
                return float(np.exp(-8.0 * d))
            except Exception:
                return 0.0
        pois_df['distance_score'] = pois_df.apply(distance_score_row, axis=1)
    except Exception:
        pois_df['distance_score'] = 0.0

    pois_df['final_score'] = (
        weights['rating']   * pois_df['rating_score'] +
        weights['reviews']  * pois_df['review_score'] +
        weights['preference'] * pois_df['preference_score'] +
        weights['distance'] * pois_df['distance_score']
    )
    ranked_df = pois_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    print("Ranking complete."); return ranked_df


def haversine_distance(lat1, lon1, lat2, lon2):
	"""Calculate distance between two points in kilometers."""
	try:
		R = 6371  # Earth's radius in kilometers
		lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
		c = 2 * math.asin(math.sqrt(a))
		return R * c
	except Exception:
		return float('inf')


def optimize_daily_routes(attractions_by_day):
	"""Optimize daily routes using greedy nearest-neighbor for each day."""
	optimized_days = []
	
	for day_attractions in attractions_by_day:
		if len(day_attractions) <= 1:
			optimized_days.append(day_attractions)
			continue
			
		# Start with highest scored attraction
		remaining = day_attractions.copy()
		optimized = [remaining.pop(0)]  # Start with best attraction
		
		while remaining:
			# Find nearest attraction to the last one in optimized route
			last_attraction = optimized[-1]
			last_lat, last_lng = last_attraction.get('lat'), last_attraction.get('lng')
			
			if last_lat is None or last_lng is None:
				# If no coordinates, just add remaining attractions
				optimized.extend(remaining)
				break
				
			nearest_idx = 0
			min_distance = float('inf')
			
			for i, attraction in enumerate(remaining):
				att_lat, att_lng = attraction.get('lat'), attraction.get('lng')
				if att_lat is not None and att_lng is not None:
					distance = haversine_distance(last_lat, last_lng, att_lat, att_lng)
					if distance < min_distance:
						min_distance = distance
						nearest_idx = i
			
			optimized.append(remaining.pop(nearest_idx))
		
		optimized_days.append(optimized)
	
	return optimized_days


def select_pois_for_itinerary(ranked_attractions_df, ranked_restaurants_df, ranked_hotels_df, num_days):
	"""Selects a subset of POIs using clustering and ranking, aiming for variety."""
	selected_attractions_by_day = [[] for _ in range(num_days)]; selected_restaurants = []; selected_hotels = []
	flat_selected_attractions = []; attractions_per_day_target = 3; restaurants_total_target = num_days * 2
	hotels_total_target = 5; selected_attraction_ids = set()

	if not ranked_attractions_df.empty:
		attractions_added = 0; attraction_pool = ranked_attractions_df.to_dict('records')
		cluster_tops = {}
		if 'cluster' in ranked_attractions_df.columns:
			try: # Handle potential empty groups if filtering is aggressive
				cluster_tops = ranked_attractions_df.loc[ranked_attractions_df.groupby('cluster')['final_score'].idxmax()].to_dict('records')
				random.shuffle(cluster_tops)
			except KeyError: # If groupby results in empty after idxmax
				 print("Warning: Could not select top picks per cluster (maybe empty groups).")
				 cluster_tops = [] # Proceed without cluster prioritization

		# Assign top cluster picks first
		for i, attraction in enumerate(cluster_tops):
			if attractions_added >= num_days * attractions_per_day_target: break
			day_index = i % num_days
			if len(selected_attractions_by_day[day_index]) < attractions_per_day_target and attraction['place_id'] not in selected_attraction_ids:
				 selected_attractions_by_day[day_index].append({k: attraction.get(k) for k in ['name', 'rating', 'types', 'place_id', 'lat', 'lng']})
				 selected_attraction_ids.add(attraction['place_id']); attractions_added += 1

		# Fill remaining spots
		current_day_fill = 0
		for attraction in attraction_pool:
			if attractions_added >= num_days * attractions_per_day_target: break
			if attraction['place_id'] not in selected_attraction_ids:
				day_index = current_day_fill % num_days
				if len(selected_attractions_by_day[day_index]) < attractions_per_day_target:
					selected_attractions_by_day[day_index].append({k: attraction.get(k) for k in ['name', 'rating', 'types', 'place_id', 'lat', 'lng']})
					selected_attraction_ids.add(attraction['place_id']); attractions_added += 1
				current_day_fill += 1
		
		# Optimize daily routes for better distance efficiency
		selected_attractions_by_day = optimize_daily_routes(selected_attractions_by_day)
		flat_selected_attractions = [item for sublist in selected_attractions_by_day for item in sublist]
		print(f"Selected {len(flat_selected_attractions)} attractions using ML pipeline with route optimization.")
	else: print("Selected 0 attractions (input DataFrame was empty).")

	if not ranked_restaurants_df.empty:
		selected_restaurants_raw = ranked_restaurants_df.head(restaurants_total_target).to_dict('records')
		selected_restaurants = [{'name': r['name'], 'rating': r['rating'], 'types': r['types'], 'place_id': r['place_id']} for r in selected_restaurants_raw]
		print(f"Selected {len(selected_restaurants)} restaurants.")
	else: print("Selected 0 restaurants."); selected_restaurants = []

	if not ranked_hotels_df.empty:
		selected_hotels_raw = ranked_hotels_df.head(hotels_total_target).to_dict('records')
		selected_hotels = [{'name': h['name'], 'rating': h['rating'], 'types': h['types'], 'place_id': h['place_id']} for h in selected_hotels_raw]
		print(f"Selected {len(selected_hotels)} hotel options.")
	else: print("Selected 0 hotels."); selected_hotels = []

	return flat_selected_attractions, selected_restaurants, selected_hotels


# --- Itinerary Generation Function ---

def generate_itinerary_ai(request_data, username):
	"""Generates structured itinerary using ML selection + Gemini arrangement."""
	if not VERTEX_AI_INITIALIZED: return {"error": "AI service unavailable. Check logs."}

	# Extract data and validate
	destination = request_data.get('destination', 'Unknown'); num_days_int = max(1, int(request_data.get('numberOfDays', 1)))
	budget_inr = max(0.0, float(request_data.get('budget', 0.0))); preferences = request_data.get('preferences', [])
	start_date = request_data.get('startDate', datetime.now().strftime('%Y-%m-%d'))
	origin_city = request_data.get('originCity', 'Delhi')
	number_of_members = max(1, int(request_data.get('numberOfMembers', 1)))
	dietary_preference = (request_data.get('dietaryPreference') or '').strip().lower()

	# Get Contextual Info
	weather_info = get_weather_forecast(destination, num_days_int)
	air_quality_info = get_air_quality(destination)
	
	# Remove flights usage per requirements
	flight_info = {"available": False, "message": "Flight option disabled"}

	# Get REAL hotel data (scale adults by members, assume 2 per room)
	rooms_needed = max(1, math.ceil(number_of_members / 2))
	hotel_info = get_hotel_options(destination, start_date, num_days_int, adults_number=number_of_members, room_number=rooms_needed)
	
	# Get train/bus options with working links
	train_links = get_irctc_links(origin_city, destination, start_date)
	bus_links = get_bus_links(origin_city, destination, start_date)
	
	# Build travel options summary (no flights)
	travel_options_summary = f"""
	Train: Search trains from {origin_city} to {destination} on {start_date}
	Bus: Search buses from {origin_city} to {destination}
	"""

	# Budget Check & INR Formatting (tripType-aware)
	warning_message = ""; budget_str = f"INR {budget_inr:,.0f}"; daily_budget_str = "N/A"
	trip_type = (request_data.get('tripType') or 'standard').strip().lower()
	min_map = {'budget': 800, 'standard': 1000, 'luxury': 2000}
	min_daily_budget_per_person_inr = min_map.get(trip_type, 1000)
	if budget_inr > 0:
		daily_budget_inr = budget_inr / num_days_int
		per_person_daily = daily_budget_inr / max(1, number_of_members)
		daily_budget_str = f"INR {daily_budget_inr:,.0f}"
		if per_person_daily < min_daily_budget_per_person_inr:
			return {"error": f"Budget too low for {number_of_members} member(s) and {num_days_int} day(s). Increase budget or reduce days.", "code": "BUDGET_TOO_LOW"}
		else:
			warning_message = f"\n\n**Budget Note:** Total {budget_str} ({daily_budget_str}/day)"
	else:
		warning_message = "\n\n**Note:** No budget specified..."

	# --- Execute ML Pipeline ---
	print("\n--- Starting ML Pipeline ---")
	raw_attractions, raw_restaurants, raw_hotels = fetch_points_of_interest(destination, preferences, dietary_preference)
	attractions_df = preprocess_pois(raw_attractions); restaurants_df = preprocess_pois(raw_restaurants); hotels_df = preprocess_pois(raw_hotels)
	clustered_attractions_df = cluster_attractions(attractions_df, num_clusters_hint=num_days_int)
	preference_weights = request_data.get('preferenceWeights') if isinstance(request_data, dict) else None
	ranked_attractions_df = rank_pois(clustered_attractions_df, preferences, preference_weights)
	ranked_restaurants_df = rank_pois(restaurants_df, preferences, preference_weights); ranked_hotels_df = rank_pois(hotels_df, [], preference_weights) # Rank hotels without prefs
	selected_attractions, selected_restaurants, selected_hotels = select_pois_for_itinerary(ranked_attractions_df, ranked_restaurants_df, ranked_hotels_df, num_days_int)
	print("--- ML Pipeline Complete ---\n")

	# Format selected POIs for the prompt
	def format_poi_for_prompt(poi_list):
		if not poi_list: return "None selected."
		return "\n".join([f"- {poi.get('name', 'N/A')} (Rating: {poi.get('rating', 'N/A')}, Types: {', '.join(poi.get('types', []))})" for poi in poi_list])
	selected_attractions_str = format_poi_for_prompt(selected_attractions)
	selected_restaurants_str = format_poi_for_prompt(selected_restaurants)
	selected_hotels_str = format_poi_for_prompt(selected_hotels)

	# --- Construct Prompt using SELECTED POIs ---
	preferences_str = ', '.join(preferences) if preferences else 'None specified'
	json_structure_example = """[{"day": integer, "narrative": "string", "activities": [{"name": "string", "search_keyword": "string"}], "restaurants": [{"name": "string", "search_keyword": "string"}], "transport_suggestion": "string", "suggested_hotel": {"name": "string", "search_keyword": "string"}(optional, only if relevant hotel found)}]""" # Added optional hotel

	# Format hotel info for prompt
	hotel_prompt_info = ""
	if hotel_info.get('available') and hotel_info.get('hotels'):
		hotel_prompt_info = "\nReal Hotel Options (with prices):\n" + "\n".join([f"- {h['description']}" for h in hotel_info['hotels']])
	else:
		hotel_prompt_info = f"\nHotel Options (from Google Maps):\n{selected_hotels_str}"

	prompt_text = f"""
	You are an expert Indian travel agent, Tripster, arranging curated places into a structured JSON itinerary.
	User: "{username}", Dest: {destination}, Origin: {origin_city}, Days: {num_days_int}, Budget: {budget_str} INR ({daily_budget_str}/day), Prefs: {preferences_str}
	Weather: {weather_info}, AQI: {air_quality_info} {warning_message}

	Travel Options from {origin_city} to {destination}:
	{travel_options_summary}

	Pre-selected POIs (optimized for route efficiency):
	Attractions:
	{selected_attractions_str}
	Restaurants:
	{selected_restaurants_str}
	{hotel_prompt_info}

	**TASK:** Create a day-by-day itinerary JSON ({num_days_int} days).
	1. Arrange "Attractions" logically (~3/day) in optimized route order. If none/few, suggest alternatives.
	2. Incorporate "Restaurants" near activities. If none/few, suggest options.
	3. Choose **one** best-fitting hotel from the hotel options (consider location, budget, and availability). Include it in the Day 1 plan within a `suggested_hotel` object like `{{"name": "Hotel Name", "search_keyword": "Hotel Name, City"}}`. If no hotels provided, omit the `suggested_hotel` field.
	4. Include travel recommendations in Day 1 narrative (suggest flight/train/bus based on budget and preferences).
	5. Response MUST be ONLY a valid JSON array following: {json_structure_example}. Ensure activities and restaurants are lists.
	6. For EVERY activity/restaurant/suggested_hotel in JSON, provide `name` and `search_keyword` for Google Maps (include city).
	7. Write `narrative` per day considering context. Include costs/transport **in INR**, respecting budget.
	8. Do NOT include text outside the JSON array `[` and `]`.

	Generate JSON:
	"""
	print(f"--- Sending Prompt to Gemini ({MODEL_NAME}) with Selected POIs & Hotels ---")

	try:
		# --- Call Gemini API ---
		if not VERTEX_AI_INITIALIZED: raise RuntimeError("Vertex AI not initialized")
		model = GenerativeModel(MODEL_NAME)
		safety_settings = { HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH, # etc.
		}
		response = model.generate_content(prompt_text, safety_settings=safety_settings)

		# --- Process Response ---
		raw_response_text = ""; finish_reason = "Unknown"
		# (Robust checking)
		if response and hasattr(response, 'candidates') and response.candidates:
			 candidate = response.candidates[0]
			 finish_reason = candidate.finish_reason.name if hasattr(candidate, 'finish_reason') and candidate.finish_reason else "UNSPECIFIED"
			 if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
				  raw_response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
		print(f"--- Received Raw Response from Gemini (Finish Reason: {finish_reason}) ---")

		# --- Attempt to Parse JSON ---
		daily_plans_structured = []
		json_string_parsed = ""
		if raw_response_text:
			try:
				# (Keep cleaning and parsing logic)
				cleaned_response = raw_response_text.strip().lstrip('```json').rstrip('```').strip()
				start_index = cleaned_response.find('['); end_index = cleaned_response.rfind(']')
				json_string_parsed = cleaned_response[start_index : end_index + 1] if start_index != -1 and end_index > start_index else cleaned_response
				daily_plans_structured = json.loads(json_string_parsed)

				# --- Generate and Add Smart Links ---
				if isinstance(daily_plans_structured, list):
					for day_plan in daily_plans_structured:
						if not isinstance(day_plan, dict): continue
						# Link Activities & Restaurants
						for item_list_key in ['activities', 'restaurants']:
							for item in day_plan.get(item_list_key, []):
								if isinstance(item, dict) and item.get('search_keyword'):
									item['links'] = generate_smart_links(item['search_keyword'])
						# Link Suggested Hotel if present
						hotel_item = day_plan.get('suggested_hotel')
						if isinstance(hotel_item, dict) and hotel_item.get('search_keyword'):
							hotel_item['links'] = generate_smart_links(hotel_item['search_keyword'])

					print("Smart links generated and added.")
				else:
					return {"error": "AI returned unexpected JSON structure (not a list)."}

			# (Keep JSONDecodeError and other processing error handling)
			except json.JSONDecodeError as json_err: return {"error": f"AI returned data in non-JSON format." }
			except Exception as process_err: return {"error": f"Failed to process plan structure: {process_err}"}
		else:
			# (Keep empty/blocked response handling)
			blocking_reasons = ['SAFETY', 'RECITATION', 'OTHER']
			if finish_reason in blocking_reasons:
				return {"error": f"Response blocked ({finish_reason})..."}
			else:
				return {"error": "AI returned an empty response."}

		# --- Compute Budget Breakdown ---
		def parse_inr(value_str):
			try:
				if isinstance(value_str, (int, float)):
					return float(value_str)
				if not value_str:
					return 0.0
				cleaned = str(value_str).replace('₹', '').replace(',', '').strip()
				return float(cleaned)
			except Exception:
				return 0.0

		# Flights disabled
		flight_cost_total = 0.0

		# Hotel cost: average of top hotels per night, scaled by nights
		hotel_cost_total = 0.0
		selected_nightly = 0.0
		if isinstance(hotel_info, dict) and hotel_info.get('available') and hotel_info.get('hotels'):
			prices = [parse_inr(h.get('price')) for h in hotel_info.get('hotels', [])]
			if prices:
				# Adjust nightly hotel spend by preference: budget=min, mid=avg, luxury=max
				hotel_pref = (request_data.get('hotelPreference') or '').strip().lower()
				if hotel_pref == 'budget':
					selected_nightly = min(prices)
				elif hotel_pref == 'luxury':
					selected_nightly = max(prices)
				else:
					selected_nightly = sum(prices) / len(prices)
				hotel_cost_total = selected_nightly * max(1, num_days_int - 1)  # nights

		# Food estimation per person per day (veg a bit lower on avg)
		food_pp_per_day = 350 if dietary_preference in ['veg', 'vegetarian'] else 450
		food_cost_total = food_pp_per_day * number_of_members * num_days_int

		# Local transport and activities estimates per person per day
		local_transport_pp_per_day = 300
		activities_pp_per_day = 500
		local_transport_total = local_transport_pp_per_day * number_of_members * num_days_int
		activities_total = activities_pp_per_day * number_of_members * num_days_int

		estimated_total = flight_cost_total + hotel_cost_total + food_cost_total + local_transport_total + activities_total

		# If user provided a budget, scale category totals depending on trip type
		scaled = {
			"food": food_cost_total,
			"local": local_transport_total,
			"activities": activities_total,
			"hotel": hotel_cost_total,
		}
		approx_total = estimated_total
		if budget_inr and budget_inr > 0 and estimated_total > 0:
			if trip_type == 'luxury':
				# Scale up or down to utilize full budget
				scale_factor = budget_inr / estimated_total
				for k in scaled:
					scaled[k] = round(scaled[k] * scale_factor, 2)
				approx_total = round(sum(scaled.values()), 2)
			elif trip_type == 'budget':
				# Budget-friendly: do not scale up; only scale down if over budget
				if estimated_total > budget_inr:
					scale_factor = budget_inr / estimated_total
					for k in scaled:
						scaled[k] = round(scaled[k] * scale_factor, 2)
					approx_total = round(sum(scaled.values()), 2)
				# else keep original estimated allocations (leave some budget unspent)
			else:
				# Standard: keep allocations as-is (middle ground)
				approx_total = round(estimated_total, 2)

		budget_breakdown = {
			"members": number_of_members,
			"dietaryPreference": dietary_preference or "",
			"approxHotelNightly": round(selected_nightly, 2),
			"foodApprox": round(scaled.get("food", food_cost_total), 2),
			"localTransportApprox": round(scaled.get("local", local_transport_total), 2),
			"activitiesApprox": round(scaled.get("activities", activities_total), 2),
			"approxTotal": round(approx_total, 2),
			"hotelPreference": (request_data.get('hotelPreference') or ''),
			"tripType": trip_type,
		}

		# --- Return Structured Data with core trip fields ---
		return {
			"dailyPlans": daily_plans_structured,
			"weatherInfo": weather_info,
			"airQualityInfo": air_quality_info,
			"travelOptions": {
				"train": {
					"available": True,
					"description": f"Search trains from {origin_city} to {destination}",
					"booking_links": train_links
				},
				"bus": {
					"available": True,
					"description": f"Search buses from {origin_city} to {destination}",
					"booking_links": bus_links
				}
			},
			"hotelBooking": hotel_info,
			# Echo back essential fields for frontend persistence/display
			"destination": destination,
			"numberOfDays": num_days_int,
			"budget": budget_inr,
			"dailyBudget": float(budget_inr / num_days_int) if budget_inr > 0 else 0.0,
			"originCity": origin_city,
			"startDate": start_date,
			"numberOfMembers": number_of_members,
			"dietaryPreference": dietary_preference,
			"budgetBreakdown": budget_breakdown
		}

	except Exception as e:
		# --- Handle API Call/General Errors ---
		# (Keep existing detailed error handling)
		print(f"ERROR: Exception during Gemini call/processing: {e}")
		error_message = f"Failed itinerary generation due to AI service error."
		# ... (Specific error checks) ...
		return {"error": error_message}