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
