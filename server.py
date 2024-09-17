from flask import Flask, render_template, request
from core import Pixie

app = Flask(__name__)
pix = Pixie()

@app.route('/')
def home():
    return render_template('pixie.html')

@app.route('/generate', methods=['POST'])
def generate():
    destination = request.form['destination']
    days = int(request.form['days'])
    date = request.form['date']
    budget = request.form['budget']
    diet = request.form['diet']
    interests = request.form.getlist('interests')
    comments = request.form['comments']
    
    prompt = pix.prompt_generator(destination, days, budget, diet, interests, comments)
    print("Prompt: ", prompt)
    
    response, entities = pix.ask_pixie(prompt)
    print("Response: ", response)
    print("Entities: ", entities)
    
    return render_template('itinerary.html', itinerary=response, entities=entities)


if __name__ == '__main__':
    app.run()