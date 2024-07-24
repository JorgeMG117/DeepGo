package test

import (
	"testing"

	"github.com/JorgeMG117/DeepGo/nn"
)

func TestMSELoss_Apply(t *testing.T) {
    loss := nn.MSELoss{}

	output := []float32{1.0, 2.0, 3.0}
	expected := []float32{1.5, 2.5, 3.5}

	got := loss.Apply(output, expected)
	want := float32(0.25) // ((1.0-1.5)^2 + (2.0-2.5)^2 + (3.0-3.5)^2) / 3 = 0.25

	if got != want {
		t.Errorf("MSELoss.Apply() = %v; want %v", got, want)
	}
}

func TestMSELoss_derivApply(t *testing.T) {
	loss := nn.MSELoss{}

	output := [][]float32{
		{1.0},
		{2.0},
		{3.0},
	}
	expected := []float32{1.5, 2.5, 3.5}

	got := loss.DerivApply(output, expected)
	want := [][]float32{
		{-0.5},
		{-0.5},
		{-0.5},
	}

	for i := range got {
		for j := range got[i] {
			if got[i][j] != want[i][j] {
				t.Errorf("MSELoss.derivApply() = %v; want %v", got, want)
			}
		}
	}
}

