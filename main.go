package main

import (
	//"fmt"
	"strconv"
	//"math"
	"encoding/csv"
	"log"
	"os"

	"github.com/JorgeMG117/DeepGo/data"
	"github.com/JorgeMG117/DeepGo/nn"
	"github.com/JorgeMG117/DeepGo/plot"
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
            nn.CreateLayer(2, 2, nn.Relu),
            nn.CreateLayer(2, 1, nn.Sigmoid),
        } }

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
            nn.CreateLayer(2, 2, nn.Relu),
            nn.CreateLayer(2, 1, nn.Sigmoid),
        } }
    
    return &data, &nn 
}

func mnist() {
}


func main() {

    // Read data
    // Create neural network
    data, nn := circle()

    plot.PlotData(data, "trainData.png")

    // Train
    nn.Train(data.Inputs, data.Targets)

    /*
    res := nn.MakePrediction(input)

    var target float32 = 0.0
    mse := math.Sqrt(float64(res-target))

    fmt.Println(res)
    fmt.Println(mse)
    */

}
