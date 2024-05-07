from app import app

app.config["SECRET_KEY"] = "123"

if __name__ == "main":
    app.run(debug=True)
