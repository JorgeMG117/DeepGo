package main

import (
	"fmt"

	"github.com/JorgeMG117/DeepGo/data"
	"github.com/JorgeMG117/DeepGo/datasets"
	"github.com/JorgeMG117/DeepGo/nn"
	"github.com/JorgeMG117/DeepGo/plot"
	"github.com/JorgeMG117/DeepGo/utils"
)

// Function to compute the confusion matrix
func confusionMatrix(predictions, labels []int, numClasses int) [][]int {
	cm := make([][]int, numClasses)
	for i := range cm {
		cm[i] = make([]int, numClasses)
	}

	for i := range predictions {
		cm[labels[i]][predictions[i]]++
	}

	return cm
}

func main() {
	Xtrain, Ytrain, Xtest, Ytest := datasets.MNIST()

	dlTrain := data.NewDataLoader(data.Data{Inputs: Xtrain, Targets: Ytrain}, 32, true)
	dlTest := data.NewDataLoader(data.Data{Inputs: Xtest, Targets: Ytest}, 32, true)

	// Plot the first 10 images
	for i := 0; i < 10; i++ {
		plot.PlotImage(Xtrain[i], fmt.Sprintf("MNIST_%d.png", i), 28, 28)
	}

	nn := nn.NN{
		Layers: []*nn.Layer{
			nn.CreateLayer(28*28, 128, nn.Relu{}),
			nn.CreateLayer(128, 64, nn.Relu{}),
			nn.CreateLayer(64, 10, nn.Softmax{}),
		},
		Lr:           0.01,
		LossFunction: nn.CrossEntropyLoss{},
	}

	n_epochs := 5
	//loss := make([]float32, 0, n_epochs)

	for epoch := 0; epoch <= n_epochs; epoch++ {

		var Ypred [][]float32
		var X []float32
		for _, batch := range dlTrain.GetBatches() {
			// Zero the Gradients
			// Forward Pass
			// Compute Loss
			// Backward Pass
			// Update Weights
			Ypred = nn.Train(batch.Inputs, batch.Targets)
			X = batch.Targets
		}
		loss := nn.LossFunction.Apply(utils.FlattenMatrixToVector(Ypred), X)
		fmt.Printf("Epoch %d/%d, Loss: %f\n", epoch+1, n_epochs, loss)
	}

	// Test
	//predictions := make([]int, 0, len(Xtest))
	for _, batch := range dlTest.GetBatches() {
		nn.MakePrediction(batch.Inputs)
		//predictions = append(predictions, utils.ArgMax(pY))
		//loss := nn.LossFunction.Apply(utils.FlattenMatrixToVector(Ypred), X)
		//fmt.Print
	}

}
