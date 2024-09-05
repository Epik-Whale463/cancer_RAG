# main1.py
from flask import Flask, render_template, request, jsonify
from llm import get_query_engine

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    query_engine = get_query_engine()
    response = query_engine.query(user_query)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)