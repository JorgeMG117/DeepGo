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

func (nn *NN) backpropagation(input [][]float32, target []float32) {
    var dE_dY [][]float32
    var dE_dX [][]float32
    var _dE_dX [][]float32
    var dE_dW [][]float32
    var err error
    /*
        Example
    Input (500, 2) (n, p)
    NN
        (2, 4)
        (4, 8)
        (8, 1)
    */

    for layer := len(nn.Layers)-1; layer >= 0; layer-- {
        fmt.Println("LAYER: ", layer)
        if layer == len(nn.Layers)-1 {
            // 500 1
            //fmt.Println("predictions size: ", len(predictions), len(predictions[0]))
            //fmt.Println("target size: ", len(target))
            dE_dY = nn.LossFunction.derivApply(nn.Layers[layer].y, target)
            fmt.Println("dE_dY: ", len(dE_dY), len(dE_dY[0]))

        } else {
            // 500 1 * 1 8 -> 500 8
            // 500 8 * 8 4 -> 500 4
            wT := utils.TransposeMatrix(nn.Layers[layer+1].weights)
            //fmt.Println("weights size: ", len(wT), len(wT[0]))
            //fmt.Println("_dedx size: ", len(_dE_dX), len(_dE_dX[0]))
            dE_dY, err = utils.MultiplyMatrices(_dE_dX, wT)
            if err != nil { fmt.Println(err) }
            fmt.Println("dE_dY: ", len(dE_dY), len(dE_dY[0]))
        }

        // Efect of each unit in error
        //500 1
        //500 8
        //500 4
        dE_dX, err = utils.ElementWiseMultiply(dE_dY, nn.Layers[layer].activationFunc.derivApply(nn.Layers[layer].y)) 
        if err != nil { fmt.Println(err) }
        fmt.Println("dE_dX: ", len(dE_dX), len(dE_dX[0]))
        _dE_dX = dE_dX

        // 8 500 * 500 1 -> 8 1
        // 4 500 * 500 8 -> 4 8
        // 2 500 * 500 4 -> 2 4
        var aux [][]float32
        if layer == 0 {
            aux = input 
        } else {
            aux = nn.Layers[layer-1].y
        }
        dE_dW, err = utils.MultiplyMatrices(utils.TransposeMatrix(aux), dE_dX) 
        if err != nil { fmt.Println(err) }
        fmt.Println("dE_dW: ", len(dE_dW), len(dE_dW[0]))

        //fmt.Println("Weight size: ", len(nn.Layers[layer].weights), len(nn.Layers[layer].weights[0]))
        //fmt.Println("dE_dW size: ", len(dE_dW), len(dE_dW[0]))
        //fmt.Println(nn.Layers[layer].weights)
        // 2 1 - 500 1
        fmt.Println("Pre weights: ", nn.Layers[layer].weights)
        fmt.Println("Pre bias: ", nn.Layers[layer].bias)
        w := utils.MultiplyMatrixByScalar(dE_dW, nn.Lr)
        b := utils.MultiplyMatrixByScalar(utils.ColumnMeans(dE_dX), nn.Lr)
        fmt.Println("weight: ", w)
        fmt.Println("bias: ", b)
        nn.Layers[layer].weights = utils.SubstractSameSize(nn.Layers[layer].weights, w)
        nn.Layers[layer].bias = utils.SubstractBias(nn.Layers[layer].bias, b)
        //nn.Layers[layer].weights = utils.SubstractSameSize(nn.Layers[layer].weights, utils.MultiplyMatrixByScalar(dE_dW, nn.Lr))
        //nn.Layers[layer].bias = utils.SubstractBias(nn.Layers[layer].bias, utils.MultiplyMatrixByScalar(utils.ColumnMeans(dE_dX), nn.Lr))
        fmt.Println("Post weights: ", nn.Layers[layer].weights)
        fmt.Println("Post bias: ", nn.Layers[layer].bias)
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
    //fmt.Println("Train data structure: ", len(input), len(input[0]))

    // Forward pass
    predictions := nn.MakePrediction(input)

    // Backward pass
    // For each neuron in layer compute ∂E/∂y
    nn.backpropagation(input, target)

    return predictions
}

