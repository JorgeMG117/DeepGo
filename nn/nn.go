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
        //fmt.Println("LAYER: ", layer)
        if layer == len(nn.Layers)-1 {
            // 500 1
            //fmt.Println("predictions size: ", len(predictions), len(predictions[0]))
            //fmt.Println("target size: ", len(target))
            dE_dY := nn.LossFunction.derivApply(predictions, target) 

            // Efect of each unit in error
            //500 1
            dE_dX, err := utils.ElementWiseMultiply(dE_dY, nn.Layers[layer].activationFunc.derivApply(predictions)) 
            if err != nil { fmt.Println(err) }
            _dE_dX = dE_dX

            // 1 500 * 500 1
            dE_dW, err = utils.MultiplyMatrices(utils.TransposeMatrix(predictions), dE_dX) 
            if err != nil { fmt.Println(err) }
        } else {
            // 500 1 * 1 8 -> 500 8
            // 500 1 * 4 8 ERROR
            wT := utils.TransposeMatrix(nn.Layers[layer+1].weights)
            //fmt.Println("weights size: ", len(wT), len(wT[0]))
            //fmt.Println("_dedx size: ", len(_dE_dX), len(_dE_dX[0]))
            dE_dY, err := utils.MultiplyMatrices(_dE_dX, wT) 
            if err != nil { fmt.Println(err) }

            // 500 8 * 500 8
            // 500 4 * 500 4
            //fmt.Println("y size: ", len(nn.Layers[layer].y), len(nn.Layers[layer].y[0]))
            //fmt.Println("dedy size: ", len(dE_dY), len(dE_dY[0]))
            dE_dX, err := utils.ElementWiseMultiply(dE_dY, nn.Layers[layer].activationFunc.derivApply(nn.Layers[layer].y)) 
            if err != nil { fmt.Println(err) }
            _dE_dX = dE_dX

            // 500 8 * 500 8
            dE_dW, err = utils.MultiplyMatrices(utils.TransposeMatrix(nn.Layers[layer].y), dE_dX) 
            if err != nil { fmt.Println(err) }
        }

        //fmt.Println("Weight size: ", len(nn.Layers[layer].weights), len(nn.Layers[layer].weights[0]))
        //fmt.Println("dE_dW size: ", len(dE_dW), len(dE_dW[0]))
        //fmt.Println(nn.Layers[layer].weights)
        // 2 1 - 500 1
        nn.Layers[layer].weights = utils.SubstractWeights(nn.Layers[layer].weights, utils.ColumnMeans(utils.MultiplyMatrixByScalar(dE_dW, nn.Lr)))
    }
}


func (nn *NN) Train(input [][]float32, target []float32) [][]float32 {
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
        (p, 2)
        (2, 4)
        (4, 8)
        (8, 1)

        First layer
        weights[2][2]
        input[500][2]

        input * weights = X[500][2]

        Second layer I have 2*4, input features * neurons, weights in total
        weights[2][4]
        input[500][2]

        input * weights = X[500][4]
        Y[500][4] = actF(X)

        Third layer
        weights[4][8]
        input[500][4]

        X[500][8] = i * w

        Fourth layer
        weights[8][1]
        input[500][8]

        X[500][1]
    */
    fmt.Println("Train data structure: ", len(input), len(input[0]))

    // Forward pass
    predictions := nn.MakePrediction(input)

    // Backward pass
    // For each neuron in layer compute ∂E/∂y
    nn.backpropagation(predictions, target)

    return predictions
}

