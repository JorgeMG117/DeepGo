package nn

import (
	"math"
)

type ActivationFunc interface {
	Apply(x [][]float32) [][]float32
	DerivApply(x [][]float32) [][]float32
}

type Identity struct{}

func (a Identity) Apply(x [][]float32) [][]float32 {
	return x
}

type Sigmoid struct{}

func (a Sigmoid) Apply(x [][]float32) [][]float32 {
	res := make([][]float32, len(x))

	for i := 0; i < len(x); i++ {
		res[i] = make([]float32, len(x[0]))
		for j := 0; j < len(x[0]); j++ {
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

type Relu struct{}

func (a Relu) Apply(x [][]float32) [][]float32 {
	res := make([][]float32, len(x))

	for i := range x {
		res[i] = make([]float32, len(x[i]))
		for j := range x[i] {
			if x[i][j] < 0 {
				res[i][j] = 0
			} else {
				res[i][j] = x[i][j]
			}
		}
	}

	return res
}

func (a Relu) DerivApply(x [][]float32) [][]float32 {
	res := make([][]float32, len(x))

	for i := range x {
		res[i] = make([]float32, len(x[i]))
		for j := range x[i] {
			if x[i][j] < 0 {
				res[i][j] = 0
			} else {
				res[i][j] = 1
			}
		}
	}

	return res
}

type Softmax struct{}

func (a Softmax) Apply(x [][]float32) [][]float32 {
	res := make([][]float32, len(x))

	for i := range x {
		res[i] = make([]float32, len(x[i]))
		var sum float32
		for j := range x[i] {
			res[i][j] = float32(math.Exp(float64(x[i][j])))
			sum += res[i][j]
		}
		for j := range x[i] {
			res[i][j] /= sum
		}
	}

	return res
}

func (a Softmax) DerivApply(x [][]float32) [][]float32 {
	res := make([][]float32, len(x))

	for i := range x {
		res[i] = make([]float32, len(x[i]))
		for j := range x[i] {
			res[i][j] = x[i][j] * (1 - x[i][j])
		}
	}

	return res
}
