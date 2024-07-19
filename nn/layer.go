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

    // Initialize the weights
    weights := make([][]float32, inputSize)
    for i := range weights {
        weights[i] = make([]float32, outputSize)
        for j := range weights[i] {
            weights[i][j] = rand.Float32()
        }
    }


    return &Layer {
        //input: make([]float32, 0, inputSize),
        //output: make([]float32, 0, outputSize),
        inputSize: inputSize,
        outputSize: outputSize,
        weights: weights,
        bias: make([]float32,outputSize),
        activationFunc: actFunc,

        //x: make([][]float32, outputSize),
        //y: make([][]float32, outputSize),
    }

}

func (l *Layer) Predict(input [][]float32) [][]float32 {
    /*
    l.x = make([][]float32, len(input))
    for i := range l.x {
        l.x[i] = make([]float32, l.outputSize)
    }
    */

    x, err := utils.MultiplyMatrices(input, l.weights)

    if err != nil { fmt.Println(err) }

    l.x = x
    //fmt.Println(l.x)
    l.y = l.activationFunc.Apply(l.x)
    //fmt.Println(l.y)
    return l.y
}

func (l *Layer) ajf() {

}


