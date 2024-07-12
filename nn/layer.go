package nn

type Layer struct {
    //input []float32
    //output []float32
    inputSize int
    outputSize int 

    weights [][]float32
    bias []float32
    activationFunc ActivationFunc
}

func CreateLayer(inputSize int, outputSize int, actFunc ActivationFunc) *Layer {
    //weights := []float32 {1.45, -0.66}
    weights := make([][]float32, outputSize)
    for i := range weights {
        weights[i] = make([]float32, inputSize)
    }

    weights[0][0] = 1.45
    weights[0][1] = -0.66

    return &Layer {
        //input: make([]float32, 0, inputSize),
        //output: make([]float32, 0, outputSize),
        inputSize: inputSize,
        outputSize: outputSize,
        weights: weights,
        bias: make([]float32,outputSize),
        activationFunc: actFunc,
    }

}

func (l *Layer) Predict(input []float32) []float32 {
    output := make([]float32, l.outputSize)
    for neuron := 0; neuron < l.outputSize; neuron++ {
        output[neuron] = l.activationFunc(dot(input, l.weights[neuron]) + l.bias[neuron])
    }

    return output
}


func dot(a, b []float32) float32 {
    var res float32
    for i := range a {
        res += a[i] * b[i]
    }
    return res
}
