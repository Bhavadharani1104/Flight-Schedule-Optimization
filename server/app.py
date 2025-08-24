# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
from pymongo import MongoClient
import os
import re
import spacy
from datetime import datetime, timedelta
import logging

# -------------------------------------------------------
# Setup logging
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "optimized_schedule_enhanced.csv")

# -------------------------------------------------------
# App / sockets / DB
# -------------------------------------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

try:
    mongo = MongoClient("mongodb://127.0.0.1:27017")
    db = mongo.flightScheduler
    users_col = db.users
    flights_col = db.flights
    # Test connection
    mongo.admin.command('ismaster')
    logger.info("MongoDB connection established")
    db_available = True
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    db = None
    db_available = False

# -------------------------------------------------------
# NLP Setup
# -------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
    nlp = None

# -------------------------------------------------------
# Helpers for time
# -------------------------------------------------------
TIME_RE = re.compile(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?$')

def time_to_mins(timestr):
    """Convert time string to minutes since midnight"""
    if not timestr or not isinstance(timestr, str):
        return None
    m = TIME_RE.match(timestr.strip())
    if not m:
        return None
    h = int(m.group(1)) % 24
    mi = int(m.group(2))
    return h * 60 + mi

def normalize_time(time_str):
    """Normalize time string to HH:MM:SS format"""
    if not time_str or pd.isna(time_str) or str(time_str).lower() == 'nan':
        return ""
    time_str = str(time_str).strip()
    if len(time_str) == 5:  # HH:MM format
        return f"{time_str}:00"
    return time_str

# -------------------------------------------------------
# Bootstrap: Just import CSV as-is
# -------------------------------------------------------
def bootstrap_db_from_csv():
    """Load flight data from CSV into MongoDB"""
    if not db_available:
        logger.error("MongoDB not available, skipping bootstrap")
        return
        
    try:
        flights_col.delete_many({})
        if not os.path.exists(CSV_PATH):
            logger.error(f"CSV not found: {CSV_PATH}")
            return

        df = pd.read_csv(CSV_PATH)
        logger.info(f"Loaded CSV with {len(df)} rows")

        # Normalize time columns
        time_columns = ["STD", "Optimal_STD", "STA", "ATA", "ATD"]
        for col in time_columns:
            if col in df.columns:
                df[col] = df[col].apply(normalize_time)

        # Add default columns if missing
        if "Runway" not in df.columns:
            df["Runway"] = "R1"
        if "FlightType" not in df.columns:
            df["FlightType"] = "Domestic"

        # Ensure numeric columns
        numeric_columns = ["Predicted_Delay", "CascadingDelay", "TotalPredictedDelay"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        df.reset_index(drop=True, inplace=True)
        df["rowId"] = df.index + 1

        # Insert data
        payload = df.to_dict(orient="records")
        flights_col.insert_many(payload)
        logger.info(f"Inserted {len(payload)} flights into database")
        
    except Exception as e:
        logger.error(f"Error in bootstrap: {e}")

# -------------------------------------------------------
# Enhanced NLP Chatbot
# -------------------------------------------------------
class FlightChatbot:
    def __init__(self, csv_path, nlp_model):
        self.csv_path = csv_path
        self.nlp = nlp_model
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load flight data from CSV"""
        try:
            if os.path.exists(self.csv_path):
                self.df = pd.read_csv(self.csv_path)
                # Normalize flight numbers for matching
                if "Flight Number" in self.df.columns:
                    self.df["Flight_Number_Upper"] = self.df["Flight Number"].str.upper()
                logger.info(f"Chatbot loaded {len(self.df)} flight records")
            else:
                logger.error(f"CSV file not found: {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading data for chatbot: {e}")

    def extract_flight_number(self, text):
        """Extract flight number using multiple methods"""
        # Method 1: Regex pattern for common flight number formats
        flight_patterns = [
            r'\b([A-Z]{1,3}[\s-]?\d{3,4})\b',  # AI2509, 6E 2301, AI-2509
            r'\b(\d[A-Z]\d{3,4})\b',           # 9W2344
            r'\b([A-Z]\d{1,2}[A-Z]?\d{2,3})\b' # SG421, G8421
        ]
        
        for pattern in flight_patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                # Clean up the match (remove spaces, hyphens)
                flight_num = re.sub(r'[\s-]', '', matches[0])
                return flight_num
        
        # Method 2: Use spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("ORG", "PRODUCT"):
                    potential_flight = re.sub(r'[^\w]', '', ent.text.upper())
                    if re.match(r'^[A-Z]{1,3}\d{3,4}$', potential_flight):
                        return potential_flight
        
        return None

    def extract_city_or_airport(self, text):
        """Extract city or airport names from text"""
        cities = []
        
        # Method 1: Look for patterns like "from [city]", "to [city]"
        patterns = [
            r'from\s+([A-Za-z\s\(\)]+?)(?:\s+to|\s*$)',
            r'to\s+([A-Za-z\s\(\)]+?)(?:\s+from|\s*$)',
            r'flights?\s+(?:from|to)\s+([A-Za-z\s\(\)]+?)(?:\s|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            cities.extend([m.strip() for m in matches if len(m.strip()) > 2])
        
        # Method 2: Use spaCy NER for locations
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC"):  # Geopolitical entity, Location
                    cities.append(ent.text.strip())
        
        return list(set(cities)) if cities else None

    def extract_delay_simulation_params(self, text):
        """Extract parameters for delay simulation from text"""
        flight_number = self.extract_flight_number(text)
        
        # Extract delay amount
        delay_match = re.search(r'(\d+)\s*min', text.lower())
        if not delay_match:
            delay_match = re.search(r'by\s+(\d+)', text.lower())
        
        delay_amount = int(delay_match.group(1)) if delay_match else 30
        
        # Extract runway
        runway_match = re.search(r'runway\s+([rR]\d+)', text, re.IGNORECASE)
        if not runway_match:
            runway_match = re.search(r'\b([rR]\d+)\b', text)
        
        runway = runway_match.group(1).upper() if runway_match else None
        
        return flight_number, delay_amount, runway

    def get_intent(self, text):
        """Determine user intent from text"""
        text_lower = text.lower()
        
        # Define intent patterns with more specific matching
        intents = {
            'what_if_simulation': ['what if', 'if i delay', 'simulate', 'scenario', 'impact of'],
            'cascading_delay': ['cascading delay', 'cascade delay', 'cascading', 'ripple effect', 'predicted cascading'],
            'delay_query': ['predicted delay', 'delay for', 'delayed', 'late', 'behind schedule'],
            'runway_query': ['runway', 'gate', 'which runway'],
            'flights_from': ['flights from', 'departures from', 'leaving from'],
            'flights_to': ['flights to', 'arrivals to', 'going to'],
            'busiest_time': ['busiest', 'busy', 'peak', 'most flights'],
            'status_query': ['status of', 'information about', 'details of', 'status', 'info about'],
            'help': ['help', 'what can you do', 'commands', 'options']
        }
        
        # Check for exact matches first (more specific)
        for intent, keywords in intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent
        
        return 'general_query'

    def answer_query(self, text):
        """Main query answering function with enhanced NLP"""
        if not self.df is not None or self.df.empty:
            return "Sorry, flight data is not available at the moment."
        
        text = text.strip()
        if not text:
            return "Please ask me something about flights."
            
        intent = self.get_intent(text)
        flight_number = self.extract_flight_number(text)
        cities = self.extract_city_or_airport(text)
        
        try:
            # Handle different intents
            if intent == 'what_if_simulation':
                return self._simulate_delay_scenario(text)
            
            elif intent == 'cascading_delay' and flight_number:
                return self._get_cascading_delay(flight_number)
            
            elif intent == 'delay_query' and flight_number:
                return self._get_predicted_delay(flight_number)
            
            elif intent == 'runway_query' and flight_number:
                return self._get_runway_info(flight_number)
            
            elif intent in ['flights_from', 'flights_to'] and cities:
                return self._get_flights_by_city(cities[0], intent)
            
            elif intent == 'busiest_time':
                return self._get_busiest_slot()
            
            elif intent == 'status_query' and flight_number:
                return self._get_flight_status(flight_number)
            
            elif intent == 'help':
                return self._get_help_message()
            
            # Fallback: try to find flight info if flight number provided
            elif flight_number:
                return self._get_flight_summary(flight_number)
            
            # General search
            else:
                return self._handle_general_query(text)
                
        except Exception as e:
            logger.error(f"Error in answer_query: {e}")
            return "Sorry, I encountered an error processing your request."

    def _get_cascading_delay(self, flight_number):
        """Get cascading delay information"""
        rows = self.df[self.df["Flight_Number_Upper"] == flight_number]
        if rows.empty:
            return f"No data found for flight {flight_number}."
        
        # Get unique entries only (remove duplicates)
        unique_rows = rows.drop_duplicates(subset=['Flight Number', 'Runway'])
        
        if len(unique_rows) == 1:
            row = unique_rows.iloc[0]
            delay = row.get('CascadingDelay', 0)
            runway = row.get('Runway', 'Unknown')
            return f"Flight {flight_number} on Runway {runway}: Cascading Delay = {delay:.1f} minutes"
        
        responses = []
        for _, row in unique_rows.iterrows():
            delay = row.get('CascadingDelay', 0)
            runway = row.get('Runway', 'Unknown')
            responses.append(f"Runway {runway}: Cascading Delay = {delay:.1f} minutes")
        
        return f"Flight {flight_number}:\n" + "\n".join(responses)

    def _get_predicted_delay(self, flight_number):
        """Get predicted delay information"""
        rows = self.df[self.df["Flight_Number_Upper"] == flight_number]
        if rows.empty:
            return f"No data found for flight {flight_number}."
        
        # Get unique entries only (remove duplicates)
        unique_rows = rows.drop_duplicates(subset=['Flight Number', 'Runway'])
        
        if len(unique_rows) == 1:
            row = unique_rows.iloc[0]
            delay = row.get('Predicted_Delay', 0)
            runway = row.get('Runway', 'Unknown')
            return f"Flight {flight_number} on Runway {runway}: Predicted Delay = {delay:.1f} minutes"
        
        responses = []
        for _, row in unique_rows.iterrows():
            delay = row.get('Predicted_Delay', 0)
            runway = row.get('Runway', 'Unknown')
            responses.append(f"Runway {runway}: Predicted Delay = {delay:.1f} minutes")
        
        return f"Flight {flight_number}:\n" + "\n".join(responses)

    def _get_runway_info(self, flight_number):
        """Get runway information for a flight"""
        rows = self.df[self.df["Flight_Number_Upper"] == flight_number]
        if rows.empty:
            return f"No data found for flight {flight_number}."
        
        runways = rows["Runway"].unique()
        return f"Flight {flight_number} is scheduled on runway(s): {', '.join(runways)}."

    def _get_flights_by_city(self, city, intent):
        """Get flights from or to a city"""
        city_lower = city.lower()
        column = "From" if intent == 'flights_from' else "To"
        
        matches = self.df[self.df[column].str.lower().str.contains(city_lower, na=False)]
        if matches.empty:
            return f"No flights found {'from' if intent == 'flights_from' else 'to'} {city.title()}."
        
        flights = matches["Flight Number"].unique()[:10]  # Limit to 10 flights
        direction = 'from' if intent == 'flights_from' else 'to'
        return f"Flights {direction} {city.title()}: {', '.join(flights)}"

    def _get_busiest_slot(self):
        """Get the busiest time slot"""
        if 'STD' not in self.df.columns:
            return "Schedule data not available."
        
        self.df['STD_hour'] = self.df['STD'].astype(str).str[:2]
        slot_counts = self.df['STD_hour'].value_counts()
        if slot_counts.empty:
            return "No schedule data available."
        
        busiest_hour = slot_counts.index[0]
        count = slot_counts.iloc[0]
        return f"The busiest time slot is {busiest_hour}:00 with {count} scheduled departures."

    def _get_flight_status(self, flight_number):
        """Get comprehensive flight status"""
        rows = self.df[self.df["Flight_Number_Upper"] == flight_number]
        if rows.empty:
            return f"No data found for flight {flight_number}."
        
        row = rows.iloc[0]
        from_loc = row.get('From', 'Unknown')
        to_loc = row.get('To', 'Unknown')
        std = row.get('STD', 'Unknown')
        runway = row.get('Runway', 'Unknown')
        pred_delay = row.get('Predicted_Delay', 0)
        
        return (f"Flight {flight_number}: {from_loc} → {to_loc}\n"
                f"Scheduled Departure: {std}\n"
                f"Runway: {runway}\n"
                f"Predicted Delay: {pred_delay:.1f} minutes")

    def _simulate_delay_scenario(self, text):
        """Simulate what-if delay scenario"""
        flight_number, delay_amount, runway = self.extract_delay_simulation_params(text)
        
        if not flight_number:
            return ("Please specify a flight number for the simulation. "
                   "Example: 'What if I delay AI2509 at runway R1 by 30 minutes?'")
        
        # Get current flight data
        flight_rows = self.df[self.df["Flight_Number_Upper"] == flight_number]
        if flight_rows.empty:
            return f"Flight {flight_number} not found in the system."
        
        # Filter by runway if specified
        if runway:
            runway_rows = flight_rows[flight_rows["Runway"].str.upper() == runway]
            if runway_rows.empty:
                available_runways = flight_rows["Runway"].unique()
                return (f"Flight {flight_number} is not scheduled on runway {runway}. "
                       f"Available runways: {', '.join(available_runways)}")
            target_row = runway_rows.iloc[0]
        else:
            target_row = flight_rows.iloc[0]
            runway = target_row["Runway"]
        
        # Current delays
        current_predicted = target_row.get('Predicted_Delay', 0)
        current_cascading = target_row.get('CascadingDelay', 0)
        
        # Simulated new delays
        new_predicted = current_predicted + delay_amount
        
        # Estimate cascading impact (simplified model)
        # Assume each additional minute of delay creates 0.3 minutes of cascading delay
        additional_cascading = delay_amount * 0.3
        new_cascading = current_cascading + additional_cascading
        
        # Find potentially affected flights (same runway, later departure)
        std_time = target_row.get('STD', '')
        same_runway_flights = self.df[
            (self.df["Runway"] == runway) & 
            (self.df["STD"] > std_time)
        ].head(5)  # Show impact on next 5 flights
        
        result = [
            f"🎯 DELAY SIMULATION FOR FLIGHT {flight_number} ON RUNWAY {runway}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"📊 CURRENT STATUS:",
            f"  • Current Predicted Delay: {current_predicted:.1f} minutes",
            f"  • Current Cascading Delay: {current_cascading:.1f} minutes",
            f"",
            f"🔄 AFTER ADDING {delay_amount} MINUTE DELAY:",
            f"  • New Predicted Delay: {new_predicted:.1f} minutes (+{delay_amount})",
            f"  • Estimated Cascading Delay: {new_cascading:.1f} minutes (+{additional_cascading:.1f})",
            f"  • Total Impact: {new_predicted + new_cascading:.1f} minutes",
            f""
        ]
        
        if not same_runway_flights.empty:
            result.extend([
                f"⚠️  POTENTIAL IMPACT ON OTHER FLIGHTS (Runway {runway}):",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ])
            
            for _, flight in same_runway_flights.iterrows():
                impact_delay = min(delay_amount * 0.8, 20)  # Diminishing impact, max 20 min
                flight_num = flight["Flight Number"]
                flight_std = flight.get("STD", "Unknown")
                current_delay = flight.get("Predicted_Delay", 0)
                new_flight_delay = current_delay + impact_delay
                
                result.append(f"  • {flight_num} (STD: {flight_std}): "
                            f"{current_delay:.1f} → {new_flight_delay:.1f} min (+{impact_delay:.1f})")
        else:
            result.append("✅ No other flights on this runway would be immediately affected.")
        
        result.extend([
            f"",
            f"💡 RECOMMENDATION:",
            f"  Consider rescheduling to runway with lower traffic or",
            f"  implementing delay recovery procedures to minimize impact."
        ])
        
        return "\n".join(result)

    def _handle_general_query(self, text):
        """Handle general queries"""
        text_lower = text.lower()
        
        if "highest delay" in text_lower or "most delayed" in text_lower:
            if 'TotalPredictedDelay' in self.df.columns:
                idx = self.df['TotalPredictedDelay'].idxmax()
                row = self.df.loc[idx]
                flight = row.get('Flight Number', 'Unknown')
                runway = row.get('Runway', 'Unknown')
                delay = row.get('TotalPredictedDelay', 0)
                return (f"Flight {flight} (Runway {runway}) has the highest "
                       f"total predicted delay: {delay:.1f} minutes")
            else:
                return "Delay information not available."
        
        return "Sorry, I couldn't understand your question. Type 'help' to see what I can do."

    def _get_flight_summary(self, flight_number):
        """Get basic flight summary"""
        rows = self.df[self.df["Flight_Number_Upper"] == flight_number]
        if rows.empty:
            return f"No data found for flight {flight_number}."
        
        row = rows.iloc[0]
        from_loc = row.get('From', 'Unknown')
        to_loc = row.get('To', 'Unknown')
        std = row.get('STD', 'Unknown')
        runway = row.get('Runway', 'Unknown')
        pred_delay = row.get('Predicted_Delay', 0)
        
        return (f"{flight_number}: {from_loc} → {to_loc}, "
                f"STD {std}, Runway {runway}, "
                f"Predicted delay {pred_delay:.1f} min")

    def _get_help_message(self):
        """Get help message"""
        return (
            "I can help you with flight information! Here are some examples:\n\n"
            "✈️ Flight Status:\n"
            "  • 'Status of AI2509'\n"
            "  • 'Information about flight 6E2301'\n\n"
            "⏰ Delays:\n"
            "  • 'Predicted delay for AI2509'\n"
            "  • 'Cascading delay for AI2509'\n\n"
            "🎯 What-If Simulations:\n"
            "  • 'What if I delay AI2509 by 30 minutes?'\n"
            "  • 'What if I delay AI2509 at runway R1 by 45 mins?'\n"
            "  • 'Simulate 20 minute delay for SG115'\n\n"
            "🛫 Routes:\n"
            "  • 'Flights from Mumbai'\n"
            "  • 'Flights to Delhi'\n\n"
            "🛤️ Operations:\n"
            "  • 'Which runway for AI2509?'\n"
            "  • 'What is the busiest time slot?'\n"
            "  • 'Which flight has the highest delay?'\n\n"
            "Just ask naturally - I understand various ways of asking!"
        )

# Initialize chatbot
chatbot = FlightChatbot(CSV_PATH, nlp) if nlp else None

# -------------------------------------------------------
# Bootstrap
# -------------------------------------------------------
bootstrap_db_from_csv()

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/")
def home():
    return "Flask + Socket.IO server is running ✅"

@app.route("/api/flights")
def flights():
    if not db_available:
        return jsonify({"error": "Database not available"}), 500
        
    try:
        flights = list(flights_col.find({}, {"_id": 0}))
        for i, f in enumerate(flights):
            if "rowId" not in f:
                f["rowId"] = i + 1
            f["id"] = f["rowId"]
            
            # Ensure numeric fields are numbers
            for col in ["Predicted_Delay", "CascadingDelay", "TotalPredictedDelay"]:
                if col in f:
                    try:
                        f[col] = float(f[col]) if f[col] is not None else 0.0
                    except (ValueError, TypeError):
                        f[col] = 0.0
                else:
                    f[col] = 0.0
            
            # Ensure times are strings
            for col in ["STD", "Optimal_STD", "STA", "ATA", "ATD"]:
                if col in f and f[col] is not None:
                    f[col] = str(f[col])
                else:
                    f[col] = ""
        
        return jsonify(flights)
    except Exception as e:
        logger.error(f"Error fetching flights: {e}")
        return jsonify({"error": "Failed to fetch flights"}), 500

# ------------------ AUTH ROUTES (plain text passwords) ------------------
@app.route("/api/register", methods=["POST"])
def register():
    if not db_available:
        return jsonify({"status": "fail", "message": "Database not available"}), 500
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            return jsonify({"status": "fail", "message": "Username and password required"}), 400
        
        if users_col.find_one({"username": username}):
            return jsonify({"status": "fail", "message": "User already exists"}), 400
        
        # Store plain password (no hashing)
        users_col.insert_one({"username": username, "password": password})
        return jsonify({"status": "success", "message": "User registered successfully"}), 201
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"status": "fail", "message": "Registration failed"}), 500

@app.route("/api/login", methods=["POST"])
def login():
    if not db_available:
        return jsonify({"status": "fail", "message": "Database not available"}), 500
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")
        
        user = users_col.find_one({"username": username})
        if user and user.get("password") == password:
            return jsonify({"status": "success", "message": "Login successful"}), 200
        
        return jsonify({"status": "fail", "message": "Invalid credentials"}), 401
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"status": "fail", "message": "Login failed"}), 500
    
@app.route("/api/update", methods=["POST"])
def update_flight():
    if not db_available:
        return jsonify({"status": "fail", "message": "Database not available"}), 500
        
    try:
        data = request.json
        row_id = data.get("rowId")
        flight_id = data.get("flightId")
        field = data.get("field")
        value = data.get("value")

        if not field:
            return jsonify({"status": "fail", "message": "Field name required"})

        # Build query
        query = {}
        if row_id is not None:
            query = {"rowId": int(row_id)}
        elif flight_id:
            query = {"Flight Number": flight_id}
        else:
            return jsonify({"status": "fail", "message": "Provide rowId or flightId"})

        doc = flights_col.find_one(query)
        if not doc:
            return jsonify({"status": "fail", "message": "Flight not found"})

        # Update specific fields with validation
        if field in ["STD", "Optimal_STD"]:
            doc[field] = normalize_time(value)
        else:
            doc[field] = value

        flights_col.update_one({"_id": doc["_id"]}, {"$set": doc})

        updated = flights_col.find_one({"_id": doc["_id"]}, {"_id": 0})
        socketio.emit("flight_updated", {"row": updated}, broadcast=True)
        
        return jsonify({"status": "success", "updatedRow": updated})
    except Exception as e:
        logger.error(f"Update error: {e}")
        return jsonify({"status": "fail", "message": "Update failed"}), 500

@app.route("/api/busiest_slots")
def busiest_slots():
    try:
        df = pd.read_csv(CSV_PATH)
        df['STD_hour'] = df['STD'].astype(str).str[:2]
        slot_counts = df.groupby('STD_hour').size().reset_index(name='count')
        return jsonify(slot_counts.to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Error getting busiest slots: {e}")
        return jsonify({"error": "Failed to get busiest slots"}), 500

@app.route("/api/high_impact_flights")
def high_impact_flights():
    try:
        df = pd.read_csv(CSV_PATH)
        if 'TotalPredictedDelay' in df.columns:
            top_flights = df.sort_values('TotalPredictedDelay', ascending=False).head(10)
            return jsonify(top_flights[['Flight Number', 'TotalPredictedDelay']].to_dict(orient="records"))
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error getting high impact flights: {e}")
        return jsonify({"error": "Failed to get high impact flights"}), 500

# ------------- Enhanced Chatbot (Socket.IO) -------------
@socketio.on("chat_message")
def on_chat_message(data):
    try:
        text = (data or {}).get("text", "").strip()
        if not text:
            emit("chat_reply", {"text": "Please ask me something about flights."})
            return
            
        if chatbot:
            reply = chatbot.answer_query(text)
        else:
            reply = "Sorry, the chatbot is not available. Please ensure spaCy is installed correctly."
        
        emit("chat_reply", {"text": reply})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        emit("chat_reply", {"text": "Sorry, I encountered an error processing your message."})

@socketio.on("connect")
def on_connect():
    logger.info("Client connected to chat")
    emit("chat_reply", {"text": "Hello! I'm your flight assistant. Type 'help' to see what I can do."})

@socketio.on("disconnect")
def on_disconnect():
    logger.info("Client disconnected from chat")

# -------------------------------------------------------
# Run
# -------------------------------------------------------
if __name__ == "__main__":
    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")