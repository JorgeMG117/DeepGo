package main

import (
	"github.com/JorgeMG117/DeepGo/data"
	"github.com/JorgeMG117/DeepGo/datasets"
	"github.com/JorgeMG117/DeepGo/nn"
	"github.com/JorgeMG117/DeepGo/plot"
	"github.com/JorgeMG117/DeepGo/utils"
    "fmt"
)

func testNN1() (*data.Data, *nn.NN) {
    aggData := data.Data{
		Inputs: [][]float32{
			{0.0, 0.0},
			{0.0, 1.0},
			{1.0, 0.0},
			{1.0, 1.0},
		},
		Targets: []float32{
			0.0,
			1.0,
			1.0,
			0.0,
		},
	}

    nn := nn.NN { 
        Layers: []*nn.Layer{
            nn.CreateLayer(2, 2, nn.Sigmoid{}),
            nn.CreateLayer(2, 1, nn.Sigmoid{}),
        },
        Lr: 0.1,
        LossFunction: nn.MSELoss{},
    }

    return &aggData, &nn
}


func mnist() {
}


func main() {

    /*
    data := &data.Data{
		Inputs: [][]float32{
			{0.1, 0.2, 0.3},
			{1.1, 1.2, 1.3},
			{2.1, 2.2, 2.3},
		},
		Targets: []float32{0.1, 0.2, 0.3},
	}

	plot.PlotHeatmap(data, "heatmap.png")
    return 
    */

    // Read data
    X := datasets.Circle()

    // Create neural network
    nn := nn.NN { 
        Layers: []*nn.Layer{
            //nn.CreateLayer(2, 2, nn.Sigmoid{}),
            nn.CreateLayer(2, 4, nn.Sigmoid{}),
            nn.CreateLayer(4, 1, nn.Sigmoid{}),
            //nn.CreateLayer(8, 1, nn.Sigmoid{}),
        },
        Lr: 0.01,
        LossFunction: nn.MSELoss{},
    }

    plot.PlotData(X, "trainData.png")

    // Train
    train_steps := 5000 
    
    loss := make([]float32, 0, train_steps)

    for iter := 0; iter <= train_steps; iter++ {

        pY := nn.Train(X.Inputs, X.Targets)

        if iter % 25 != 0 { continue }

        loss = append(loss, nn.LossFunction.Apply(utils.FlattenMatrixToVector(pY), X.Targets))

        fmt.Println("Loss: ", loss)
        //fmt.Println("pY: ", pY)

        if iter == 1 { 
            return 
        }

        /*
        const numData int = 50 
        _x0 := utils.Linspace(-1.5, 1.5, numData)
        _x1 := utils.Linspace(-1.5, 1.5, numData)
        var _Y [numData][numData]float32
        //data.CreateData()

        predictions := data.Data {
            Inputs:  make([][]float32, 0, 50*50),
            Targets: make([]float32, 0, 50*50),
        }

        for i0, x0 := range _x0 {
            for i1, x1 := range _x1 {
                input := make([][]float32, 1)
                input[0] = make([]float32, 2) 
                input[0][0] = x0 
                input[0][1] = x1

                _Y[i0][i1] = nn.MakePrediction(input)[0][0]

                predictions.Inputs = append(predictions.Inputs, input[0])
			    predictions.Targets = append(predictions.Targets, _Y[i0][i1])


            }
        }

        iterS := fmt.Sprint(iter)
        plot.PlotPredictions(X, &predictions, "predictedData" + iterS + ".png")
        */

    }

    const numData int = 100 
    _x0 := utils.Linspace(-1.5, 1.5, numData)
    _x1 := utils.Linspace(-1.5, 1.5, numData)
    var _Y [numData][numData]float32
    //data.CreateData()

    predictions := data.Data {
        Inputs:  make([][]float32, 0, 50*50),
        Targets: make([]float32, 0, 50*50),
    }

    for i0, x0 := range _x0 {
        for i1, x1 := range _x1 {
            input := make([][]float32, 1)
            input[0] = make([]float32, 2) 
            input[0][0] = x0 
            input[0][1] = x1

            _Y[i0][i1] = nn.MakePrediction(input)[0][0]

            predictions.Inputs = append(predictions.Inputs, input[0])
            predictions.Targets = append(predictions.Targets, _Y[i0][i1])


        }
    }

    plot.PlotPredictions(X, &predictions, "predictedData.png")

    fmt.Println("Loss: ", loss)
    plot.PlotLost(loss, "loss.png")
}
