from app import create_app

app = create_app()

if __name__ == '__main__':
    # Runs the Flask development server
    # Debug=True allows auto-reloading on code changes and provides better error messages
    app.run(debug=True)