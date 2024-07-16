package plot

import (
	"fmt"

	"github.com/JorgeMG117/DeepGo/data"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

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
