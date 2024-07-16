package nn

import (
    "math"
)

type ActivationFunc func(float32) float32

func Identity(x float32) float32 {
    return x
}

func Sigmoid(x float32) float32 {
    return float32(1 / (1 + math.Exp(float64(-x))))
}

func sigmoidDerivative1(x float32) float32 {
    return 0.0
}

func Relu(x float32) float32 {
    return float32(math.Max(0.0, float64(x)))
}
