package test

import (
	"math"
	"testing"

	"github.com/JorgeMG117/DeepGo/nn"
)

const epsilon = 1e-6

func almostEqual(a, b float32) bool {
	return math.Abs(float64(a)-float64(b)) <= epsilon
}

func TestIdentity_Apply(t *testing.T) {
	identity := nn.Identity{}

	input := [][]float32{
		{1.0, 2.0},
		{3.0, 4.0},
	}

	got := identity.Apply(input)
	want := input

	for i := range got {
		for j := range got[i] {
			if !almostEqual(got[i][j], want[i][j]) {
				t.Errorf("Identity.Apply() = %v; want %v", got, want)
				return
			}
		}
	}
}

func TestSigmoid_Apply(t *testing.T) {
	sigmoid := nn.Sigmoid{}

	input := [][]float32{
		{0.0, 2.0},
		{3.0, -4.0},
	}

	got := sigmoid.Apply(input)
	want := [][]float32{
		{0.5, float32(1 / (1 + math.Exp(-2)))},
		{float32(1 / (1 + math.Exp(-3))), float32(1 / (1 + math.Exp(4)))},
	}

	for i := range got {
		for j := range got[i] {
			if !almostEqual(got[i][j], want[i][j]) {
				t.Errorf("Sigmoid.Apply() = %v; want %v", got, want)
				return
			}
		}
	}
}

func TestSigmoid_derivApply(t *testing.T) {
	sigmoid := nn.Sigmoid{}

	input := [][]float32{
		{0.0, 0.5},
		{0.7, 0.9},
	}

	got := sigmoid.DerivApply(input)
	want := [][]float32{
		{0.0, 0.25},
		{0.21, 0.09},
	}

	for i := range got {
		for j := range got[i] {
			if !almostEqual(got[i][j], want[i][j]) {
				t.Errorf("Sigmoid.derivApply() = %v; want %v", got, want)
				return
			}
		}
	}
}

/* Uncomment this function and the corresponding test case once the Relu function is implemented
func Relu(x [][]float32) [][]float32 {
	for i := range x {
		for j := range x[i] {
			x[i][j] = float32(math.Max(0, float64(x[i][j])))
		}
	}
	return x
}

func TestRelu_Apply(t *testing.T) {
	input := [][]float32{
		{1.0, -2.0},
		{-3.0, 4.0},
	}

	got := Relu(input)
	want := [][]float32{
		{1.0, 0.0},
		{0.0, 4.0},
	}

	for i := range got {
		for j := range got[i] {
			if !almostEqual(got[i][j], want[i][j]) {
				t.Errorf("Relu.Apply() = %v; want %v", got, want)
				return
			}
		}
	}
}
*/


