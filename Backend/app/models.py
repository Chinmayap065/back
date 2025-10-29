from . import mongo # Import the mongo instance from __init__.py

# We don't need a formal class definition like in Spring Data.
# Flask-PyMongo interacts directly with MongoDB collections.
# We will define helper functions or interact directly via mongo.db.users

# Example of how you might fetch a user later in your routes/services:
# def find_user_by_username(username):
#     return mongo.db.users.find_one({"username": username})

# Example of how you might save a user:
# def save_user(user_data):
#     return mongo.db.users.insert_one(user_data)

# It's good practice to create indexes for fields you search often
def create_user_indexes():
    """Creates unique indexes on username and email fields if they don't exist."""
    try:
        # Check existing indexes (optional, create_index handles duplicates)
        # existing_indexes = mongo.db.users.index_information()
        # print("Existing indexes:", existing_indexes)

        mongo.db.users.create_index("username", unique=True)
        mongo.db.users.create_index("email", unique=True)
        print("User indexes created successfully (or already exist).")
    except Exception as e:
        # Don't crash the app if index creation fails (e.g., during testing)
        print(f"Warning: Error creating user indexes (might already exist or DB unavailable): {e}")

# Note: We call create_user_indexes() from __init__.py within the app context