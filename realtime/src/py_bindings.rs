use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::RealtimeDf;

#[pyclass(name = "RealtimeDf", unsendable)]
pub struct PyRealtimeDf {
    inner: RealtimeDf,
}

#[pymethods]
impl PyRealtimeDf {
    #[new]
    fn new(channels: usize, atten_lim: f32) -> PyResult<Self> {
        let inner = RealtimeDf::new(channels, atten_lim).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create RealtimeDf: {}",
                e
            ))
        })?;
        Ok(Self { inner })
    }

    /// Process a chunk of audio frames
    /// Input shape: (channels, samples)
    /// Output shape: (channels, samples)
    /// Note: samples must be equal to hop_size
    #[pyo3(text_signature = "(self, input_array)")]
    fn process_frames<'py>(
        &mut self,
        py: Python<'py>,
        input_array: PyReadonlyArray2<f32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let input = input_array.as_array();
        let output = self.inner.process_frames(input.to_owned()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to process frames: {}",
                e
            ))
        })?;
        Ok(output.into_pyarray(py))
    }

    #[getter]
    fn get_hop_size(&self) -> usize {
        self.inner.get_hop_size()
    }

    #[getter]
    fn get_sample_rate(&self) -> usize {
        self.inner.get_sample_rate()
    }

    #[getter]
    fn get_channels(&self) -> usize {
        self.inner.get_channels()
    }
}

#[pymodule]
pub fn deep_filter_rt(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRealtimeDf>()?;
    Ok(())
}
