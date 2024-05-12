import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib

# Initialize Dash app
app = dash.Dash(__name__)

# Load trained model
model = joblib.load('logreg_model.pkl')

# Define layout of the app
app.layout = html.Div([
    html.H1("Spam Detection"),
    dcc.Textarea(
        id='text-input',
        placeholder='Enter text here...',
        style={'width': '100%', 'height': 50}
    ),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.P("ham means the message is safe."),
    html.P("spam means the message is unsafe."),
    html.Div(id='output')
])

# Define callback to handle prediction
@app.callback(
    Output('output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('text-input', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        # Make prediction using the trained model
        prediction = model.predict([value])[0]
        # Map prediction to label
        label = 'spam' if prediction == 1 else 'ham'
        return f'Your message is {label}.'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
