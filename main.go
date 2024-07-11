package main

import(
    "fmt"
    "math"
)

type ActivationFunc func(float32) float32

func Sigmoid(x float32) float32 {
    return float32(1 / (1 + math.Exp(float64(-x))))
}



type Layer struct {
    //input []float32
    //output []float32
    inputSize int
    outputSize int 

    weights []float32
    bias float32
    activationFunc ActivationFunc
}

func CreateLayer(inputSize int, outputSize int, actFunc ActivationFunc) *Layer {
    weights := []float32 {1.45, -0.66}
    return &Layer {
        //input: make([]float32, 0, inputSize),
        //output: make([]float32, 0, outputSize),
        inputSize: inputSize,
        outputSize: outputSize,
        weights: weights,
        bias: 0.0,
        activationFunc: actFunc,
    }

}

func (l *Layer) Predict(input []float32) []float32 {
    return []float32{l.activationFunc(dot(input, l.weights) + l.bias)}
}


func dot(a, b []float32) float32 {
    var res float32
    for i := range a {
        res += a[i] * b[i]
    }
    return res
}

type NN struct {
    layers []*Layer
}

func (nn *NN) MakePrediction(input []float32) float32 {
    for _, layer := range nn.layers {
        input = layer.Predict(input)
    }
    return input[0]
}


func main() {

    // Read data
    input := []float32{1.66, 1.56}

    // Create neural network
    layer := CreateLayer(2, 1, Sigmoid)

    nn := NN { 
        layers: []*Layer{
            layer,
        } }

    res := nn.MakePrediction(input)

    fmt.Println(res)

    // Train
}
