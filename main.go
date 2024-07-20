package main

import (
	"fmt"
	"strconv"
	//"math"
	"encoding/csv"
	"log"
	"os"

	"github.com/JorgeMG117/DeepGo/data"
	"github.com/JorgeMG117/DeepGo/nn"
	"github.com/JorgeMG117/DeepGo/plot"
	"github.com/JorgeMG117/DeepGo/utils"
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

func circle() (*data.Data, *nn.NN) {
    // Read data
    file, err := os.Open("datasets/circle_data.csv")
	if err != nil {
		log.Fatal("Error opening the CSV file:", err)
	}
	defer file.Close()

    reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1 // Allows variable number of fields per record

	// Read and print all records
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal("Error reading CSV records:", err)
	}

	// Iterate over records
    var data data.Data
    data.Targets = make([]float32, len(records)-1)
    data.Inputs = make([][]float32, len(records)-1)

    for index, record := range records[1:] {
    
        //fmt.Println("Record", index, ":", record)

        x1, err := strconv.ParseFloat(record[0], 64)
        if err != nil {
            log.Fatalf("Error parsing X1 on record %d: %v", index, err)
        }
        x2, err := strconv.ParseFloat(record[1], 64)
        if err != nil {
            log.Fatalf("Error parsing X2 on record %d: %v", index, err)
        }
        label, err := strconv.Atoi(record[2])
        if err != nil {
            log.Fatalf("Error parsing Label on record %d: %v", index, err)
        }


        data.Inputs[index] = make([]float32, 2)
        data.Inputs[index][0] = float32(x1)
        data.Inputs[index][1] = float32(x2)
        data.Targets[index] = float32(label)

        //fmt.Println(data.Inputs[index][0], data.Targets[index])
	}

    nn := nn.NN { 
        Layers: []*nn.Layer{
            nn.CreateLayer(2, 2, nn.Sigmoid{}),
            nn.CreateLayer(2, 4, nn.Sigmoid{}),
            nn.CreateLayer(4, 8, nn.Sigmoid{}),
            nn.CreateLayer(8, 1, nn.Sigmoid{}),
        },
        Lr: 0.1,
        LossFunction: nn.MSELoss{},
    }
    
    return &data, &nn 
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
    // Create neural network
    X, nn := circle()

    plot.PlotData(X, "trainData.png")

    // Train
    train_steps := 1000
    
    loss := make([]float32, train_steps)

    for iter := 0; iter <= train_steps; iter++ {

        pY := nn.Train(X.Inputs, X.Targets)

        if iter % 25 != 0 { continue }

        loss = append(loss, nn.LossFunction.Apply(utils.FlattenMatrixToVector(pY), X.Targets))
        fmt.Println("Loss: ", loss)
        fmt.Println("pY: ", pY)

        _x0 := utils.Linspace(-1.5, 1.5, 10)
        _x1 := utils.Linspace(-1.5, 1.5, 10)
        var _Y [10][10]float32
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
            //fmt.Println(predictions)
            //return
        }

        iterS := fmt.Sprint(iter)
        plot.PlotPredictions(X, &predictions, "predictedData" + iterS + ".png")





    }

    /*
    res := nn.MakePrediction(input)

    var target float32 = 0.0
    mse := math.Sqrt(float64(res-target))

    fmt.Println(res)
    fmt.Println(mse)
    */

}
