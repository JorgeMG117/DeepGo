package datasets

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"github.com/JorgeMG117/DeepGo/data"
)


func Circle() *data.Data {
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

    return &data
}

