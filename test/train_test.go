package test

import (
	"fmt"
	"os"
	"testing"
    "math"

	"github.com/JorgeMG117/DeepGo/datasets"
	"github.com/JorgeMG117/DeepGo/nn"
)

func TestMain(m *testing.M) {
    // Get the current working directory
    cwd, err := os.Getwd()
    if err != nil {
        panic(err)
    }

    // Change to the main project directory
    if err := os.Chdir(".."); err != nil {
        panic(err)
    }

    // Run the tests
    code := m.Run()

    // Reset the working directory
    if err := os.Chdir(cwd); err != nil {
        panic(err)
    }

    // Exit with the appropriate code
    os.Exit(code)
}

// CompareWeightsAndBiases compares the weights and biases with the expected values
func CompareWeightsAndBiases(expectedWeightsAndBiases, actualWeightsAndBiases  []struct {
	W [][]float32
	b []float32
}, tolerance float32) {
    for layer := range expectedWeightsAndBiases {
        fmt.Printf("Layer %d Weights Comparison:\n", layer)
        //Weight
        for i := range expectedWeightsAndBiases[layer].W {
            for j := range expectedWeightsAndBiases[layer].W[i] {
                diff := math.Abs(float64(expectedWeightsAndBiases[layer].W[i][j] - actualWeightsAndBiases[layer].W[i][j]))
                if diff > float64(tolerance) {
                    fmt.Printf("  Weight difference at [%d][%d]: %.6f (expected) vs %.6f (actual), difference: %.6f\n",
                        i, j, expectedWeightsAndBiases[layer].W[i][j], actualWeightsAndBiases[layer].W[i][j], diff)
                }
            }
        }
        //Bias
        fmt.Printf("Layer %d Biases Comparison:\n", layer)
        for i := range expectedWeightsAndBiases[layer].b {
            diff := math.Abs(float64(expectedWeightsAndBiases[layer].b[i] - actualWeightsAndBiases[layer].b[i]))
            if diff > float64(tolerance) {
                fmt.Printf("  Bias difference at [%d]: %.6f (expected) vs %.6f (actual), difference: %.6f\n",
                    i, expectedWeightsAndBiases[layer].b[i], actualWeightsAndBiases[layer].b[i], diff)
            }
        }
    }
}

func TestTrain(t *testing.T) {
    // Read data
    X := datasets.Circle()

    // Create neural network
    nn := nn.NN { 
        Layers: []*nn.Layer{
            nn.CreateLayer(2, 4, nn.Sigmoid{}),
            nn.CreateLayer(4, 8, nn.Sigmoid{}),
            nn.CreateLayer(8, 1, nn.Sigmoid{}),
        },
        Lr: 0.01,
        LossFunction: nn.MSELoss{},
    }

    // Set weights and bias
    weightsAndBiases := []struct {
		W [][]float32
		b []float32
	}{
		{
			W: [][]float32{
				{0.81434652, -0.62796093, -0.59111353, 0.48546698},
				{0.80760497, -0.08360876, -0.9396614, 0.4061006},
			},
			b: []float32{-0.7861409, 0.1265443, 0.86613917, -0.14280303},
		},
		{
			W: [][]float32{
				{-0.70645876, -0.65716068, -0.9084427, 0.13880791, -0.17437521, -0.4825444, -0.30553342, -0.45736191},
				{0.04933959, 0.0936449, 0.74106735, -0.18827566, -0.31801874, 0.53047357, -0.11539899, 0.69321849},
				{-0.04918416, -0.55343834, 0.75991654, 0.85738387, -0.39686879, -0.58143884, -0.95679225, 0.94020378},
				{0.54560789, 0.56745135, 0.65950558, 0.89902577, -0.10021978, 0.92309948, 0.67503689, 0.8865549},
			},
			b: []float32{-0.57653548, 0.78964064, -0.13616693, 0.23611745, 0.4514434, -0.96944117, 0.90166843, -0.59398517},
		},
		{
			W: [][]float32{
				{-0.57952052},
				{0.87866831},
				{-0.90259633},
				{0.39189187},
				{0.63426782},
				{-0.17413464},
				{-0.1874133},
				{0.68601979},
			},
			b: []float32{0.54583245},
		},
	}

    for i, layer := range nn.Layers {
        layer.SetWeightsAndBiases(weightsAndBiases[i].W, weightsAndBiases[i].b)

        layer.PrintWeightsAndBiases()
	}

    // Train
    nn.Train(X.Inputs, X.Targets)

    // Check weight and bias have the right values
    // Expected weights and biases
	expectedWeightsAndBiases := []struct {
		W [][]float32
		b []float32
	}{
		{
			W: [][]float32{
				{0.7980403, -0.61795799, -0.57848683, 0.49432767},
				{0.78980134, -0.0718334, -0.93345834, 0.41390238},
			},
			b: []float32{-0.78623288, 0.12664191, 0.86615005, -0.14285871},
		},
		{
			W: [][]float32{
				{-7.50306369e-01, -6.05167527e-01, -9.43202361e-01, 1.66590488e-01, -1.25328009e-01, -4.89393693e-01, -3.20386791e-01, -4.11731299e-01},
				{6.07157591e-02, 8.06127432e-02, 7.41740223e-01, -1.97537247e-01, -3.31602061e-01, 5.31983900e-01, -1.10761806e-01, 6.83606207e-01},
				{-6.84769910e-04, -6.11420137e-01, 8.02817887e-01, 8.27414198e-01, -4.50898845e-01, -5.73675938e-01, -9.40801694e-01, 8.89020662e-01},
				{5.32385357e-01, 5.83708851e-01, 6.58977022e-01, 9.08082764e-01, -8.49150365e-02, 9.21874145e-01, 6.69289472e-01, 8.98457819e-01},
			},
			b: []float32{-0.57634974, 0.78942214, -0.13594361, 0.23600041, 0.45123723, -0.96940484, 0.90172119, -0.5941875},
		},
		{
			W: [][]float32{
				{-0.51987697},
				{0.76586747},
				{-0.97324705},
				{0.31599694},
				{0.58103874},
				{-0.03467582},
				{-0.2476223},
				{0.67896099},
			},
			b: []float32{0.54547246},
		},
	}

    //var actualWeightsAndBiases []struct { W[][]float32; b []float32 }
    actualWeightsAndBiases := make([]struct { W[][]float32; b []float32 }, 3)
    for i, layer := range nn.Layers {
        actualWeightsAndBiases[i].W, actualWeightsAndBiases[i].b = layer.GetWeightsAndBiases() 
	}

    CompareWeightsAndBiases(expectedWeightsAndBiases, actualWeightsAndBiases, 1e-5)
}
