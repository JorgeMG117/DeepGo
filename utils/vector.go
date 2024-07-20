package utils

func Linspace(start, stop float32, num int) []float32 {
	if num < 2 {
		return []float32{start}
	}

	step := (stop - start) / float32(num-1)
	linspace := make([]float32, num)

	for i := 0; i < num; i++ {
		linspace[i] = start + float32(i)*step
	}

	return linspace
}
