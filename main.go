package main

import (
	"fmt"
	"math"

	"github.com/JorgeMG117/DeepGo/nn"
)


func main() {

    // Read data
    input := []float32{2, 1.5}

    // Create neural network
    layer1 := nn.CreateLayer(2, 1, nn.Sigmoid)
    //layer2 := nn.CreateLayer(2, 1, nn.Identity)

    nn := nn.NN { 
        Layers: []*nn.Layer{
            layer1,
            //layer2,
        } }

    nn.Train()

    res := nn.MakePrediction(input)

    var target float32 = 0.0
    mse := math.Sqrt(float64(res-target))

    fmt.Println(res)
    fmt.Println(mse)

    // Train
}
