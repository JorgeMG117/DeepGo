package nn

import "fmt"


type NN struct {
    Layers []*Layer
    //lr // Learning rate
    LossFunction LossFunct 
}

func (nn *NN) MakePrediction(input [][]float32) []float32 {
    for _, layer := range nn.Layers {
        input = layer.Predict(input)
    }
    //return input[0]
    return nil
}


func (nn *NN) Train(input [][]float32, target []float32) {
    /*
    // Forward pass
    // For every layer get x and y

    // Being x the total input ,x_j, to unit j is linear function of the outputs, y_i, of the units that are connected to j and of the weights, w_ji, on this connections
    // x_j = Σ (from i=1 to n) of y_i * w_ji

    // Being y_j the output of a unit, which is a non-linear function of its total input
    // when using Sigmoid activation function
    // y_j = 1 / (1 + exp(-x_j))

    Example
    Input (500, 2) (n, p)
    NN
        (p, 4)
        (4, 2)
        (2, 1)

        In the first layer I have 2*4, input features * neurons, weights in total
        weights[4][2]
        input[2][500]

        weights * input = X[4][500]
        Y[4][500] = actF(X)

        Second layer
        weights[2][4]
        input[4][500]

        X[2][500] = w*i

        Third layer
        weights[1][2]
        input[2][500]

        X[1][500]
    */
    fmt.Println("Train data structure: ", len(input), len(input[0]))

    // Forward pass
    nn.MakePrediction(input)

    // Backward pass
    // For each neuron in layer compute ∂E/∂y
    /*
    dE_dY := nn.LossFunction.derivApply(predictions, target) 
    for layer := len(nn.Layers)-1; layer >= 0; layer-- {
        dE_dX := dE_dY * nn.Layers[i].ActivationFunc.derivApply(predictions) 
        dE_dW := dE_dY * 
    }
    */
}

func (nn *NN) backpropagation() {
}
