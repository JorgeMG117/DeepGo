package plot

import (
	"fmt"

	"github.com/JorgeMG117/DeepGo/data"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type HeatMap struct {
	data *data.Data
}

func (h HeatMap) Dims() (c, r int) {
	return len(h.data.Inputs[0]), len(h.data.Inputs)
}

func (h HeatMap) Z(c, r int) float64 {
	return float64(h.data.Targets[r])
}

func (h HeatMap) X(c int) float64 {
	return float64(h.data.Inputs[0][c])
}

func (h HeatMap) Y(r int) float64 {
	return float64(h.data.Inputs[r][0])
}

func PlotHeatmap(data *data.Data, file string) {
	plt := plot.New()

	// Create a heatmap and add it to the plot
	h := &HeatMap{data: data}
	heat := plotter.NewHeatMap(h, palette.Heat(12, 1))

	plt.Add(heat)

	// Add a color bar to the right of the plot
	//cb := plotter.ColorBar{ColorMap: palette.Heat(12, 1) }
	//plt.Add(cb)
	plt.X.Label.Text = "X"
	plt.Y.Label.Text = "Y"

	if err := plt.Save(4*vg.Inch, 4*vg.Inch, file); err != nil {
		fmt.Errorf("could not save plot: %v", err)
	}
	fmt.Println("Data saved in:", file)
}

func PlotPredictions(X *data.Data, predictions *data.Data, file string) {
    n := 4
    pts := make([]plotter.XYer, n)
    
    var l1 plotter.XYs
    var l2 plotter.XYs

    for i, v := range X.Inputs {
        if X.Targets[i] == 0 {
            l1 = append(l1, plotter.XY{ X: float64(v[0]), Y: float64(v[1]) })
        } else {
            l2 = append(l2, plotter.XY{ X: float64(v[0]), Y: float64(v[1]) })
        }
    }
    pts[0] = l1
    pts[1] = l2

    var l3 plotter.XYs
    var l4 plotter.XYs

    for i, v := range predictions.Inputs {
        if predictions.Targets[i] <= 0.5 {
            l3 = append(l3, plotter.XY{ X: float64(v[0]), Y: float64(v[1]) })
        } else {
            l4 = append(l4, plotter.XY{ X: float64(v[0]), Y: float64(v[1]) })
        }
    }
    pts[2] = l3
    pts[3] = l4

	plt := plot.New()

	// Add the points that are summarized by the error points.
	//plotutil.AddScatters(plt, pts[0], pts[1], pts[2], pts[3])
	plotutil.AddScatters(plt, pts[2], pts[3])


    /*
	// Create a heatmap and add it to the plot
    aux1 := &data.Data{
		Inputs: [][]float32{
			{0.1, 0.2},
			{1.1, 1.2},
			{2.1, 2.2},
		},
		Targets: []float32{0.1, 0.2, 0.3},
	}
	h := &HeatMap{data: aux1}
	heat := plotter.NewHeatMap(h, palette.Heat(12, 1))

	plt.Add(heat)
    */


	if err := plt.Save(4*vg.Inch, 4*vg.Inch, file); err != nil {
		fmt.Printf("could not save plot: %v", err)
	}
	fmt.Println("Data saved in:", file)
}

func PlotData(data *data.Data, file string) {
    /*
    n, m := 5, 10
	pts := make([]plotter.XYer, n)
	for i := range pts {
		xys := make(plotter.XYs, m)
		pts[i] = xys
		center := float64(i)
		for j := range xys {
			xys[j].X = center + (rand.Float64() - 0.5)
			xys[j].Y = center + (rand.Float64() - 0.5)
		}
	}
    */

    n := 2
    pts := make([]plotter.XYer, n)
    var l1 plotter.XYs
    var l2 plotter.XYs

    for i, v := range data.Inputs {
        if data.Targets[i] == 0 {
            l1 = append(l1, plotter.XY{ X: float64(v[0]), Y: float64(v[1]) })
        } else {
            l2 = append(l2, plotter.XY{ X: float64(v[0]), Y: float64(v[1]) })
        }
    }
    pts[0] = l1
    pts[1] = l2

    //fmt.Println(pts)
    //fmt.Println(l1)

	plt := plot.New()

	// Add the points that are summarized by the error points.
	plotutil.AddScatters(plt, pts[0], pts[1])

	plt.Save(4*vg.Inch, 4*vg.Inch, file)
    fmt.Println("Data saved in: ", file)
}

func PlotLost(loss []float32, file string) {
    p := plot.New()

    p.Title.Text = "Loss Plot"
	p.X.Label.Text = "Iteration"
	p.Y.Label.Text = "Loss"

    pts := make(plotter.XYs, len(loss))
	for i := range pts {
        pts[i].X = float64(i)
		pts[i].Y = float64(loss[i])
	}

    err := plotutil.AddLines(p, "Loss", pts)
	if err != nil {
		panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, file); err != nil {
		panic(err)
	}
    fmt.Println("Loss saved in: ", file)

}
