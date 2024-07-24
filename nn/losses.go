package nn

//cost function, loss function

type LossFunct interface {
    Apply(output []float32, expected []float32) float32
    DerivApply(output [][]float32, expected []float32) [][]float32
}

type MSELoss struct {}

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
