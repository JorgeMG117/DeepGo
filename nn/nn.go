package nn


type NN struct {
    Layers []*Layer
    //lr // Learning rate
}

func (nn *NN) MakePrediction(input []float32) float32 {
    for _, layer := range nn.Layers {
        input = layer.Predict(input)
    }
    return input[0]
}


func (nn *NN) Train() {
}
