package nn

import (
	"fmt"
	"math"
)

//cost function, loss function

type LossFunct interface {
	Apply(output []float32, expected []float32) float32
	DerivApply(output [][]float32, expected []float32) [][]float32
}

type MSELoss struct{}

func (l MSELoss) Apply(output []float32, expected []float32) float32 {
	if len(output) != len(expected) {
		panic("output and expected must have the same length")
	}

	var res float64
	for i := 0; i < len(output); i++ {
		diff := float64(output[i]) - float64(expected[i])
		res += diff * diff
	}

	return float32(res / float64(len(output)))
}

func (l MSELoss) DerivApply(output [][]float32, expected []float32) [][]float32 {
	if len(output) != len(expected) {
		panic("output and expected must have the same length")
	}

	res := make([][]float32, len(output))
	for i := 0; i < len(output); i++ {
		if len(output[i]) != 1 {
			panic("output must be a matrix with single-column rows")
		}
		res[i] = make([]float32, 1)
		//res[i][0] = 2 * (output[i][0] - expected[i])
		res[i][0] = output[i][0] - expected[i]
	}

	return res
}

type CrossEntropyLoss struct{}

func (l CrossEntropyLoss) Apply(output []float32, expected []float32) float32 {
	if len(output) != len(expected) {
		panic("output and expected must have the same length")
	}

	var res float64
	for i := 0; i < len(output); i++ {
		if expected[i] == 1 {
			res += -math.Log(float64(output[i]))
		} else {
			res += -math.Log(1 - float64(output[i]))
		}
	}

	return float32(res / float64(len(output)))
}

func (l CrossEntropyLoss) DerivApply(output [][]float32, expected []float32) [][]float32 {
	fmt.Println("Shape of the output: ", len(output), len(output[0]))
	fmt.Println("Shape of the expected: ", len(expected))
	if len(output) != len(expected) {
		panic("output and expected must have the same length")
	}

	res := make([][]float32, len(output))
	for i := 0; i < len(output); i++ {
		if len(output[i]) != len(expected) {
			panic("each output and expected must have the same length")
		}
		res[i] = make([]float32, len(output[i]))
		for j := 0; j < len(output[i]); j++ {
			res[i][j] = output[i][j] - expected[j]
		}
	}

	fmt.Println("Shape of the derivative: ", len(res), len(res[0]))

	return res
}
