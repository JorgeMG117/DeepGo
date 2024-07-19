package nn

import (
	"fmt"

	"github.com/JorgeMG117/DeepGo/utils"
)


type NN struct {
    Layers []*Layer
    Lr float32 // Learning rate
    LossFunction LossFunct 
}

func (nn *NN) MakePrediction(input [][]float32) [][]float32 {
    for _, layer := range nn.Layers {
        // fmt.Println("Layer input size: ", len(input))
        input = layer.Predict(input)
    }
    //fmt.Println(len(input[0])) 1
    //return input[0]
    return input 
}

func (nn *NN) backpropagation(predictions [][]float32, target []float32) {

    var _dE_dX [][]float32
    var dE_dW [][]float32
    for layer := len(nn.Layers)-1; layer >= 0; layer-- {
        if layer == len(nn.Layers)-1 {
            // 500 1
            //fmt.Println("predictions size: ", len(predictions), len(predictions[0]))
            //fmt.Println("target size: ", len(target))
            dE_dY := nn.LossFunction.derivApply(predictions, target) 

            // Efect of each unit in error
            predictionsT := utils.TransposeMatrix(predictions)
            //500 1 * 1 500
            dE_dX, err := utils.MultiplyMatrices(dE_dY, nn.Layers[layer].activationFunc.derivApply(predictionsT)) 
            if err != nil { fmt.Println(err) }

            _dE_dX = utils.ReduceMatrixToOneColumn(dE_dX)
            // 500 1 * 500 1
            dE_dW, err = utils.MultiplyMatrices(dE_dX, predictions) 
            if err != nil { fmt.Println(err) }
        } else {
            // 500 1 * 2 1 -> 500 2
            // 500 2 * 4 2
            wT := utils.TransposeMatrix(nn.Layers[layer+1].weights)
            fmt.Println("weights size: ", len(wT), len(wT[0]))
            fmt.Println("_dedx size: ", len(_dE_dX), len(_dE_dX[0]))
            dE_dY, err := utils.MultiplyMatrices(_dE_dX, wT) 
            if err != nil { fmt.Println(err) }

            // 500 2 * 500 2
            // 500 4 * 500 4
            yT := utils.TransposeMatrix(nn.Layers[layer].activationFunc.derivApply(nn.Layers[layer].y))
            dE_dX, err := utils.MultiplyMatrices(dE_dY, yT) 
            if err != nil { fmt.Println(err) }

            //_dE_dX = dE_dX
            _dE_dX = utils.ReduceMatrixToOneColumn(dE_dX)
            // 500 2
            dE_dW, err = utils.MultiplyMatrices(dE_dX, nn.Layers[layer].y) 
            if err != nil { fmt.Println(err) }
        }

        fmt.Println("Weight size: ", len(nn.Layers[layer].weights), len(nn.Layers[layer].weights[0]))
        fmt.Println("dE_dW size: ", len(dE_dW), len(dE_dW[0]))
        fmt.Println(nn.Layers[layer].weights)
        // 2 1 - 500 1
        nn.Layers[layer].weights = utils.SubstractWeights(nn.Layers[layer].weights, utils.ColumnMeans(utils.MultiplyMatrixByScalar(dE_dW, nn.Lr)))
    }
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
            nn.CreateLayer(2, 2, nn.Sigmoid{}),
            nn.CreateLayer(2, 4, nn.Sigmoid{}),
            nn.CreateLayer(4, 8, nn.Sigmoid{}),
            nn.CreateLayer(8, 1, nn.Sigmoid{}),
        (p, 4)
        (4, 2)
        (2, 1)

        In the first layer I have 2*4, input features * neurons, weights in total
        weights[2][4]
        input[500][2]

        input * weights = X[500][4]
        Y[500][4] = actF(X)

        Second layer
        weights[4][2]
        input[500][4]

        X[500][2] = w*i

        Third layer
        weights[2][1]
        input[500][2]

        X[500][1]
    */
    fmt.Println("Train data structure: ", len(input), len(input[0]))

    // Forward pass
    predictions := nn.MakePrediction(input)

    // Backward pass
    // For each neuron in layer compute ∂E/∂y
    nn.backpropagation(predictions, target)
}

