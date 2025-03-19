package data

import (
	"math/rand"
)

type InputType int

type Data struct {
	Inputs  [][]float32
	Targets []float32
}

type DataLoader struct {
	Dataset   Data
	BatchSize int
	Shuffle   bool
}

func NewDataLoader(dataset Data, batchSize int, shuffle bool) *DataLoader {
	dl := &DataLoader{
		Dataset:   dataset,
		BatchSize: batchSize,
		Shuffle:   shuffle,
	}

	if shuffle {
		indices := rand.Perm(len(dl.Dataset.Targets))
		shuffledInputs := make([][]float32, len(dl.Dataset.Inputs))
		shuffledTargets := make([]float32, len(dl.Dataset.Targets))
		for i, idx := range indices {
			shuffledInputs[i] = dl.Dataset.Inputs[idx]
			shuffledTargets[i] = dl.Dataset.Targets[idx]
		}
		dl.Dataset.Inputs = shuffledInputs
		dl.Dataset.Targets = shuffledTargets
	}

	return dl
}

// GetBatches returns the dataset divided into batches according to the BatchSize.
func (dl *DataLoader) GetBatches() []Data {
	var batches []Data
	dataLen := len(dl.Dataset.Targets)
	for i := 0; i < dataLen; i += dl.BatchSize {
		end := i + dl.BatchSize
		if end > dataLen {
			end = dataLen
		}
		batchInputs := dl.Dataset.Inputs[i:end]
		batchTargets := dl.Dataset.Targets[i:end]
		batches = append(batches, Data{Inputs: batchInputs, Targets: batchTargets})
	}
	return batches
}

func CreateData(numFeatures int, numData int) *Data {
	data := Data{
		Inputs:  make([][]float32, numData),
		Targets: make([]float32, numData),
	}

	for i := range data.Inputs {
		data.Inputs[i] = make([]float32, numFeatures)
	}

	return &data
}
