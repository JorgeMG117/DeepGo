package utils

import (
    "fmt"
)

func MultiplyMatrices(a [][]float32, b [][]float32) ([][]float32, error) {
    if len(a[0]) != len(b) {
        return nil, fmt.Errorf("cannot multiply, incompatible dimensions")
    }

    result := make([][]float32, len(a))
    for i := range result {
        result[i] = make([]float32, len(b[0]))
        for j := range result[i] {
            for k := range b {
                result[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    return result, nil
}

func MultiplyMatrixVector(matrix [][]float32, vector []float32) ([]float32, error) {
    if len(matrix) == 0 || len(vector) == 0 {
        return nil, fmt.Errorf("matrix or vector cannot be empty")
    }
    
    // Check if all rows in the matrix have the same number of columns as the size of the vector
    for _, row := range matrix {
        if len(row) != len(vector) {
            return nil, fmt.Errorf("the number of columns in the matrix must match the size of the vector")
        }
    }
    
    result := make([]float32, len(matrix))
    
    for i, row := range matrix {
        for j, value := range row {
            result[i] += value * vector[j]
        }
    }
    
    return result, nil
}

func TransposeMatrix(matrix [][]float32) [][]float32 {
    if len(matrix) == 0 {
        return nil // Return nil for empty input to avoid panics on indexing.
    }

    // Determine the number of columns in the first row assuming a non-jagged matrix.
    numRows := len(matrix)
    numCols := len(matrix[0])

    // Create a new matrix with the dimensions swapped.
    transposed := make([][]float32, numCols)
    for i := range transposed {
        transposed[i] = make([]float32, numRows)
        for j := range transposed[i] {
            transposed[i][j] = matrix[j][i]
        }
    }

    return transposed
}

func FlattenMatrixToVector(matrix [][]float32) []float32 {
	if len(matrix) == 0 {
		return nil // Return nil if the matrix is empty.
	}

	// Initialize a slice to hold the flattened matrix
	vector := make([]float32, len(matrix))

	// Flatten the matrix
	for i, row := range matrix {
		if len(row) == 0 {
			continue // Skip empty rows, though this should not happen in a properly formatted 500x1 matrix.
		}
		vector[i] = row[0] // Assume each row has exactly one element, since the matrix is 500x1.
	}

	return vector
}

func ReduceMatrixToOneColumn(matrix [][]float32) [][]float32 {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return nil // Return nil for empty input to avoid panics on indexing.
    }

    // Initialize the resulting 500x1 matrix
    result := make([][]float32, len(matrix))
    for i := range result {
        result[i] = make([]float32, 1) // Each row has one element
    }

    // Compute the average of each row
    for i, row := range matrix {
        sum := 0.0
        for _, value := range row {
            sum += float64(value)
        }
        result[i][0] = float32(sum / float64(len(row))) // Assuming all rows are non-empty and have equal length
    }

    return result
}


func MultiplyMatrixByScalar(matrix [][]float32, scalar float32) [][]float32 {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return nil// Handle empty matrix or empty rows gracefully
	}

	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] *= scalar
		}
	}

    return matrix
}

func ColumnMeans(matrix [][]float32) [][]float32 {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return nil // Handle empty matrix or columns
	}

	numCols := len(matrix[0])
	numRows := len(matrix)

	// Initialize a result matrix with 1 row and the same number of columns as the input matrix
	result := make([][]float32, 1)
	result[0] = make([]float32, numCols)

	// Sum values in each column
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			result[0][j] += matrix[i][j]
		}
	}

	// Divide each sum by the number of rows to get the mean
	for j := 0; j < numCols; j++ {
		result[0][j] /= float32(numRows)
	}

	return result
}


func SubstractBias(bias []float32, dError [][]float32) []float32 {
	if len(dError) == 0 || len(dError[0]) == 0 {
		return nil // Handle empty matrix or columns
	}

	for i := 0; i < len(bias); i++ {
        bias[i] -= dError[0][i]
	}

    return bias
}

func SubstractSameSize(x1 [][]float32, x2 [][]float32) [][]float32 {

	numCols := len(x1[0]) //1
	numRows := len(x1)//2

	result := make([][]float32, numRows)

    //fmt.Println(dError)
	for i := 0; i < numRows; i++ {
	    result[i] = make([]float32, numCols)
		for j := 0; j < numCols; j++ {
            result[i][j] = x1[i][j] - x2[i][j]
		}
	}

    return result 
}

func SumBias(x [][]float32, bias []float32) [][]float32 {
    //fmt.Println(bias)
	if len(x) == 0 || len(x[0]) == 0 {
		return nil // Handle empty matrix or columns
	}

	numCols := len(x[0])
	numRows := len(x)

	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
            x[i][j] += bias[j] 
		}
	}

    return x 
}


func ElementWiseMultiply(matrix1, matrix2 [][]float32) ([][]float32, error) {
	// Check if both matrices have the same dimensions
	if len(matrix1) == 0 || len(matrix2) == 0 || len(matrix1) != len(matrix2) || len(matrix1[0]) != len(matrix2[0]) {
		return nil, fmt.Errorf("matrices must have the same dimensions")
	}

	// Initialize the result matrix
	result := make([][]float32, len(matrix1))
	for i := range result {
		result[i] = make([]float32, len(matrix1[0]))
	}

	// Perform element-wise multiplication
	for i := range matrix1 {
		for j := range matrix1[0] {
			result[i][j] = matrix1[i][j] * matrix2[i][j]
		}
	}

	return result, nil
}
