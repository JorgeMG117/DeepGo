package main

import (
	"github.com/JorgeMG117/DeepGo/datasets"
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
	//Xtrain, Ytrain, Xtest, Ytest := datasets.MNIST()
	datasets.MNIST()

	/*
		nn := nn.NN{
			Layers: []*nn.Layer{
				nn.CreateLayer(28*28, 128, nn.Relu{}),
				nn.CreateLayer(128, 64, nn.Relu{}),
				nn.CreateLayer(64, 10, nn.Sigmoid{}),
			},
			Lr:           0.01,
			LossFunction: nn.MSELoss{},
		}


		   def forward(self, x):
		       x = x.view(-1, 28*28)  # Flatten the input
		       x = F.relu(self.fc1(x))
		       x = F.relu(self.fc2(x))
		       x = self.fc3(x)
		       return F.log_softmax(x, dim=1)
	*/
}
