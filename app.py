import dash
from dash import html

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Hi guys!"),
    html.P("Testing Render for our app deployment.")
])

# Expose server for deployment (important for Render/Heroku)
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)