package datasets

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
)

// Constants for the MNIST dataset
const (
	trainImagesFile = "train-images-idx3-ubyte.gz"
	trainLabelsFile = "train-labels-idx1-ubyte.gz"
	testImagesFile  = "t10k-images-idx3-ubyte.gz"
	testLabelsFile  = "t10k-labels-idx1-ubyte.gz"
)

// ReadMNISTImages reads the MNIST images from the given file
func ReadMNISTImages(filename string) ([][]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var magic, numImages, numRows, numCols int32
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.BigEndian, &numRows); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.BigEndian, &numCols); err != nil {
		return nil, err
	}

	if magic != 2051 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}

	images := make([][]float32, numImages)
	for i := range images {
		images[i] = make([]float32, numRows*numCols)
		for j := range images[i] {
			var pixel uint8
			if err := binary.Read(reader, binary.BigEndian, &pixel); err != nil {
				return nil, err
			}
			images[i][j] = float32(pixel) / 255.0
		}
	}

	return images, nil
}

// ReadMNISTLabels reads the MNIST labels from the given file
func ReadMNISTLabels(filename string) ([]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	var magic, numLabels int32
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.BigEndian, &numLabels); err != nil {
		return nil, err
	}

	if magic != 2049 {
		return nil, fmt.Errorf("invalid magic number: %d", magic)
	}

	uint8Labels := make([]uint8, numLabels)
	if err := binary.Read(reader, binary.BigEndian, &uint8Labels); err != nil {
		return nil, err
	}

	labels := make([]float32, numLabels)
	for i, label := range uint8Labels {
		labels[i] = float32(label)
	}

	return labels, nil
}

func MNIST() ([][]float32, []float32, [][]float32, []float32) {
	dataDir := "datasets/MNIST/raw"
	trainImagesPath := filepath.Join(dataDir, trainImagesFile)
	trainLabelsPath := filepath.Join(dataDir, trainLabelsFile)
	testImagesPath := filepath.Join(dataDir, testImagesFile)
	testLabelsPath := filepath.Join(dataDir, testLabelsFile)

	trainImages, err := ReadMNISTImages(trainImagesPath)
	if err != nil {
		fmt.Println("Error reading train images:", err)
		return nil, nil, nil, nil
	}

	trainLabels, err := ReadMNISTLabels(trainLabelsPath)
	if err != nil {
		fmt.Println("Error reading train labels:", err)
		return nil, nil, nil, nil
	}

	testImages, err := ReadMNISTImages(testImagesPath)
	if err != nil {
		fmt.Println("Error reading test images:", err)
		return nil, nil, nil, nil
	}

	testLabels, err := ReadMNISTLabels(testLabelsPath)
	if err != nil {
		fmt.Println("Error reading test labels:", err)
		return nil, nil, nil, nil
	}

	fmt.Printf("Train Images: %d, Train Labels: %d\n", len(trainImages), len(trainLabels))
	fmt.Printf("Test Images: %d, Test Labels: %d\n", len(testImages), len(testLabels))

	fmt.Println("Train Image Shape:", len(trainImages[0]))

	return trainImages, trainLabels, testImages, testLabels
}
