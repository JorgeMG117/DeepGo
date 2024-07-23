package nn

import (
    "math"
)

type ActivationFunc interface {
    Apply(x [][]float32) [][]float32
    derivApply(x [][]float32) [][]float32
}

type Identity struct {}

func (a Identity) Apply(x [][]float32) [][]float32 {
    return x
}


type Sigmoid struct {}

func (a Sigmoid) Apply(x [][]float32) [][]float32 {
    for i := 0; i < len(x); i++  {
        for j := 0; j < len(x[0]); j++  {
            x[i][j] = float32(1 / (1 + math.Exp(float64(-x[i][j]))))
        }
    }
    return x
}

func (a Sigmoid) derivApply(x [][]float32) [][]float32 {
    for i := range x {
        for j := range x[i] {
            x[i][j] = x[i][j] * (1 - x[i][j])
        }
    }
    return x
}

/*
func Relu(x [][]float32) [][]float32 {
    return float32(math.Max(0.0, float64(x)))
}
*/
