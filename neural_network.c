#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//define the neural network structure
#define INPUT_NODES 2
#define HIDDEN_NODES 2
#define OUTPUT_NODES 1

// Sigmoid activation function
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x){
    return x * (1.0 - x);
}

// Randomly initialize weights between -1 and 1
double random_weight(){
    return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

// Function to initialize weights
void initialize_weights(double weights[][HIDDEN_NODES], double hidden_weights[HIDDEN_NODES][OUTPUT_NODES]){
    for(int i = 0; i < INPUT_NODES; i++){
        for(int j = 0; j < HIDDEN_NODES; j++){
            weights[i][j] = random_weight();
        }
    }

    for(int i = 0; i < HIDDEN_NODES; i++){
        for(int j = 0; j < OUTPUT_NODES; j++){
            hidden_weights[i][j] = random_weight();
        }
    }
}

int main(){
    // Initialize weights for the nueral network
    double weights[INPUT_NODES][HIDDEN_NODES];
    double hidden_weights[HIDDEN_NODES][OUTPUT_NODES];

    initialize_weights(weights, hidden_weights);

    // Training data (for AND gate)    
    double input_data[4][2] = {{0,0},{0,1}, {1,0}, {1,1}};
    double output_data[4][1] = {{0}, {0}, {0}, {1}};

    // Initialize hidden layer and output layer arrays
    double hidden_layer[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    double learning_rate = 0.5;
    int epochs = 10000;

    // Training loop
    for(int epoch = 0; epoch < epochs; epoch++){
        for (int i = 0; i < 4; i++){
            // Feedforward pass

            // Calculate hidden layer activations
            for(int j = 0; j < HIDDEN_NODES; j++){
                hidden_layer[j] = 0;
                for (int k = 0;k < INPUT_NODES; k++){
                    hidden_layer[j] += input_data[i][k] * weights[k][j];
                }
                hidden_layer[j] = sigmoid(hidden_layer[j]);
            }

            // Calculate output layer activations
            for (int j = 0; j< OUTPUT_NODES; j++){
                output_layer[j] = 0;
                for (int k = 0; k < HIDDEN_NODES; k++){
                    output_layer[j] += hidden_layer[k] * hidden_weights[k][j];
                }
                output_layer[j] = sigmoid(output_layer[j]);
            }

            //Blackpropagation step

            // Output layer error
            double output_error[OUTPUT_NODES];
            for (int j = 0; j < OUTPUT_NODES; j++){
                output_error[j] = output_data[i][j] - output_layer[j];
            }

            // Hidden layer error
            double hidden_error[HIDDEN_NODES];
            for (int j = 0; j < HIDDEN_NODES; j++){
                hidden_error[j] = 0;
                for (int k = 0; k < HIDDEN_NODES; k++){
                    weights[j][k] += learning_rate * hidden_error[k] * input_data[i][j];
                }
                hidden_error[j] *= sigmoid_derivative(hidden_layer[j]);
            }

            // update hidden to output weight
            for (int j = 0; j < HIDDEN_NODES; j++){
                for (int k = 0; k < OUTPUT_NODES; k++){
                    hidden_weights[j][k] += learning_rate * hidden_error[k] * input_data[i][j];
                }
            }
        }
    }

    // Test  the neural network after training
    for (int i = 0; i < 4; i++){
        // Feedforward pass forr testing
        for (int j = 0; j < HIDDEN_NODES; j++){
            hidden_layer[j];
            for (int k = 0; k < INPUT_NODES; k++){
                hidden_layer[j] += input_data[i][k] * weights[k][j];
            }
            hidden_layer[j] = sigmoid(hidden_layer[j]);
        }

        for (int j = 0; j < OUTPUT_NODES; j++){
            output_layer[j] = 0;
            for (int k = 0; k < HIDDEN_NODES; j++){
                output_layer[j] += hidden_layer[k] * hidden_weights[k][j];
            }
            output_layer[j] = sigmoid(output_layer[j]);
        }

        printf("Input: [%d, %d], Predicted Output: %.2f\n", (int)input_data[i][0], (int)input_data[i][1], output_layer[0]);
    }

    return 0;
}
