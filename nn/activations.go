package nn

import (
    "math"
)

type ActivationFunc interface {
    Apply(x [][]float32) [][]float32
    DerivApply(x [][]float32) [][]float32
}

type Identity struct {}

func (a Identity) Apply(x [][]float32) [][]float32 {
    return x
}


type Sigmoid struct {}

func (a Sigmoid) Apply(x [][]float32) [][]float32 {
    res := make([][]float32, len(x))

    for i := 0; i < len(x); i++  {
        res[i] = make([]float32, len(x[0]))
        for j := 0; j < len(x[0]); j++  {
            res[i][j] = float32(1 / (1 + math.Exp(float64(-x[i][j]))))
        }
    }
    return res
}

func (a Sigmoid) DerivApply(x [][]float32) [][]float32 {
    res := make([][]float32, len(x))

    for i := range x {
        res[i] = make([]float32, len(x[i]))
        for j := range x[i] {
            res[i][j] = x[i][j] * (1 - x[i][j])
        }
    }
    return res
}

/*
func Relu(x [][]float32) [][]float32 {
    return float32(math.Max(0.0, float64(x)))
}
*/
