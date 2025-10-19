#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        // softmax_regression_epoch_cpp(
        // 	static_cast<const float*>(X.request().ptr),
        //     static_cast<const unsigned char*>(y.request().ptr),
        //     static_cast<float*>(theta.request().ptr),
        //     X.request().shape[0],
        //     X.request().shape[1],
        //     theta.request().shape[1],
        //     lr,
        //     batch
        //    );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
