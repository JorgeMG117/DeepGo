package nn

import (
	"fmt"
	"math/rand"

	"github.com/JorgeMG117/DeepGo/utils"
)

type Layer struct {
    x [][]float32
    y [][]float32

    inputSize int   // Input data size or Previous number of neurons
    outputSize int //Number of neurons 

    weights [][]float32
    bias []float32
    activationFunc ActivationFunc
}

func CreateLayer(inputSize int, outputSize int, actFunc ActivationFunc) *Layer {

    // Initialize the weights and bias
    weights := make([][]float32, inputSize)
    for i := range weights {
        weights[i] = make([]float32, outputSize)
        for j := range weights[i] {
            weights[i][j] = rand.Float32()*2-1
        }
    }

    bias := make([]float32, outputSize)
    for i := range bias {
        bias[i] = rand.Float32()*2-1
    }


    return &Layer {
        //input: make([]float32, 0, inputSize),
        //output: make([]float32, 0, outputSize),
        inputSize: inputSize,
        outputSize: outputSize,
        weights: weights,
        bias: bias,
        activationFunc: actFunc,

        //x: make([][]float32, outputSize),
        //y: make([][]float32, outputSize),
    }

}

func (l *Layer) Predict(input [][]float32) [][]float32 {
    x, err := utils.MultiplyMatrices(input, l.weights)

    if err != nil { fmt.Println(err) }

    l.x = utils.SumBias(x, l.bias)
    //fmt.Println(l.x)
    l.y = l.activationFunc.Apply(l.x)
    //fmt.Println(l.y)
    return l.y
}


func (l *Layer) SetWeightsAndBiases(W [][]float32, b []float32) {
    l.weights = W
    l.bias = b
}

func (l *Layer) PrintWeightsAndBiases() {
    fmt.Println("Weights:")
    for _, row := range l.weights {
        fmt.Println(row)
    }
    fmt.Println("Biases:")
    fmt.Println(l.bias)
}

func (l *Layer) GetWeightsAndBiases() ([][]float32, []float32) {
    return l.weights, l.bias
}
