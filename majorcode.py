import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import numpy as np
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('majorfrontend.html')

def AVOA_optimization(objective_function, bounds, iterations=50, population_size=10):
    population = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(population_size, len(bounds)))

    for iteration in range(iterations):
        for i in range(population_size):
            r1, r2, r3 = np.random.randint(0, population_size, size=3)
            random_vector = 0.5 * (population[r1] + population[r2]) + 0.5 * (population[r3] - population[i])
            population[i] = np.clip(population[i] + random_vector, bounds[:, 0], bounds[:, 1])

    best_index = np.argmin([objective_function(individual) for individual in population])
    best_solution = population[best_index]

    return best_solution

''' def SMO(objective_function, bounds, population_size=10, iterations=50, alpha=1.0, beta=1.0, gamma=1.0):
    def initialize_population(bounds, population_size):
        return np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(population_size, len(bounds)))

    def clamp(position, bounds):
        return np.clip(position, bounds[:, 0], bounds[:, 1])

    population = initialize_population(bounds, population_size)
    fitness_values = np.array([objective_function(individual) for individual in population])

    for iteration in range(iterations):
        for i in range(population_size):
            r1, r2, r3 = np.random.randint(0, population_size, size=3)
            random_vector = alpha * (population[r1] - population[r2]) + beta * (population[r3] - population[i])
            population[i] = clamp(population[i] + gamma * random_vector, bounds)

        new_fitness_values = np.array([objective_function(individual) for individual in population])

        for i in range(population_size):
            if new_fitness_values[i] < fitness_values[i]:
                population[i] = clamp(population[i] + gamma * (population[i] - population[np.argmin(fitness_values)]), bounds)

        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        best_fitness = fitness_values[best_index]

    return best_solution'''

# Load the Amazon Food Reviews dataset
data = pd.read_csv('/content/drive/MyDrive/Reviews.csv', nrows=568455)
data = data.dropna(subset=['Summary', 'Score'])
data['Sentiment'] = data['Score'].apply(lambda score: 'positive' if score > 3 else ('negative' if score < 3 else 'neutral'))

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data['Summary'], data['Sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Define the objective function for optimization
def objective_function(params):
    alpha, beta, gamma = params

    alpha = max(alpha, 1.0e-10)

    with ignore_warnings(category=ConvergenceWarning):
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(train_vectors, train_labels)

    predictions = classifier.predict(test_vectors)

    accuracy = accuracy_score(test_labels, predictions)

    return -accuracy

# Set up the bounds for the parameters
bounds = [(50, 200), (10, 50), (5,15)]

# Run African Vultures Optimization (AVOA)
avoa_best_params = AVOA_optimization(objective_function, np.array(bounds), iterations=50, population_size=10)

# Run Spider Monkey Optimization (SMO)
#smo_best_params = SMO(objective_function, np.array(bounds), population_size=10, iterations=50)

# Choose the best parameters from both optimizations
best_params = avoa_best_params # if objective_function(avoa_best_params) > objective_function(smo_best_params) else smo_best_params

# Use the best hyperparameters to train the final model
alpha, beta, gamma = best_params
print(avoa_best_params)
final_classifier = MultinomialNB(alpha=max(alpha, 1.0e-10))
final_classifier.fit(train_vectors, train_labels)
print(train_labels)
# Predict on the test set using the final model
final_predictions = final_classifier.predict(test_vectors)

# Evaluate the final performance on the test set
final_accuracy = accuracy_score(test_labels, final_predictions)
print(f'Final Accuracy on Test Set: {final_accuracy:.4f}\n')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    review_text = request.json['review_text']
    input_vector = vectorizer.transform([review_text])
    predicted_sentiment = classifier.predict(input_vector)[0]
    return jsonify({'sentiment': predicted_sentiment})

if __name__ == '_main_':
    app.run(debug=True)
# User input 
'''
user_input = input("Enter the text for sentiment classification: ")

# Use the trained classifier and vectorizer for sentiment classification
input_vector = vectorizer.transform([user_input])
predicted_sentiment = final_classifier.predict(input_vector)[0]

# Display the result
print(f"Predicted Sentiment: {predicted_sentiment}")
'''