package data

type Data struct {
	Inputs  [][]float32
	Targets []float32
}

func CreateData(numFeatures int, numData int) *Data {
    data := Data {
        Inputs:  make([][]float32, numData),
        Targets: make([]float32, numData),
    }

    for i := range data.Inputs {
        data.Inputs[i] = make([]float32, numFeatures) 
    }

    return &data
}
