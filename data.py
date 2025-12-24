# load data
import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# call dataset
dataset = loadtxt('data.csv', delimiter=',') # values separated by , 

# split into input (X) & output (y) variables
    # dataset[rows, columns]
X = dataset[:, 0:8]
y = dataset[:, 8]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

'''
First Layer: 12 nodes, reLU activation function
Second Layer: 8 nodes, reLU activation function
Output layer: 1 node, sigmoid activation function

ReLU: hidden layers, 0 to infinity
Sigmoid: final output, 0-1 (binary output)
'''

# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# fit model
'''
training occurs over epochs, each epoch split into batches
    epoch : one pass through all the rows
    batch : samples considered for weights
'''

model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)

# evaluate
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)


# make predictions
predictions = model.predict(X) 

# === user input ===
def get_user_prediction():
    print('=' * 50)
    print("RISK PREDICTOR")
    print("Please enter your values for each factor:\n")

    # feature names
    print("1. Number of pregnancies:")
    pregnancies = float(input("   → "))

    print("\n2. Glucose level:")
    glucose = float(input("   → "))

    print("\n3. Blood pressure (mm Hg):")
    blood_pressure = float(input("   → "))

    print("\n4. Skin thickness (mm):")
    skin_thickness = float(input("   → "))

    print("\n5. Insulin level (mu U/ml):")
    insulin = float(input("   → "))

    print("\n6. BMI (weight in kg/(height in m)^2):")
    bmi = float(input("   → "))

    print("\n7. Diabetes pedigree function:")
    dpf = float(input("   → "))

    print("\n8. Age (years):")
    age = float(input("   → "))

    user_input = []

    # Create input array
    user_data = np.array([[pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age]])

# Make prediction
    probability = model.predict(user_data, verbose=0)[0][0]

# --------------------------------------------
# Function to use in Flask
# --------------------------------------------

def predict(user_values):
    """
    user_values: list of 8 floats
    returns: (probability, test_accuracy)
    """
    user_array = np.array([user_values])
    probability = model.predict(user_array, verbose=0)[0][0]
    return probability, test_accuracy
