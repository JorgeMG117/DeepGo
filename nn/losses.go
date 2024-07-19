package nn

import (
	"math"
)

//cost function, loss function

type LossFunct interface {
    Apply(output []float32, expected []float32) float32
    derivApply(output [][]float32, expected []float32) [][]float32
}

type MSELoss struct {}

func (l MSELoss) Apply(output []float32, expected []float32) float32 {
    var res float64 = 0.0
    for i := 0; i < len(output); i++ {
        res += math.Pow(float64(output[i]) - float64(expected[i]), 2)
    }

    return float32(res/2)
}

func (l MSELoss) derivApply(output [][]float32, expected []float32) [][]float32 {
    res := make([][]float32, len(output))

    chEnd := make([]chan float32, len(output))
    for i := range chEnd {
        chEnd[i] = make(chan float32)
    }

    for i := 0; i < len(output); i++ {
        go func(idx int) {
            chEnd[idx] <- output[idx][0] - expected[idx]
        }(i)
    }

    for i := 0; i < len(output); i++ {
        res[i] = make([]float32, 1)
        res[i][0] = <-chEnd[i]
    }

    return res
}
