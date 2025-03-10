use std::time::Instant;

use df::tract::*;
use ndarray::prelude::*;

mod py_bindings;

pub struct RealtimeDf {
    model: DfTract,
    channels: usize,
    sample_rate: usize,
    hop_size: usize,
}

impl RealtimeDf {
    pub fn new(channels: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let df_params = DfParams::default();
        let r_params = RuntimeParams::default_with_ch(channels);
        let model = DfTract::new(df_params, &r_params)?;

        let sample_rate = model.sr;
        let hop_size = model.hop_size;

        Ok(Self {
            model,
            channels,
            sample_rate,
            hop_size,
        })
    }

    /// Process a chunk of audio frames
    /// Input shape: (channels, samples)
    /// Output shape: (channels, samples)
    /// Note: samples must be equal to hop_size
    pub fn process_frames(
        &mut self,
        input: Array2<f32>,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        assert_eq!(input.shape()[0], self.channels, "Input channels mismatch");
        assert_eq!(
            input.shape()[1],
            self.hop_size,
            "Input samples must equal hop_size"
        );

        let mut output = Array2::zeros((self.channels, self.hop_size));
        self.model.process(input.view(), output.view_mut())?;

        Ok(output)
    }

    pub fn get_hop_size(&self) -> usize {
        self.hop_size
    }

    pub fn get_sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn get_channels(&self) -> usize {
        self.channels
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example usage
    let mut df = RealtimeDf::new(2)?; // Stereo

    // Create dummy input data
    let input = Array2::zeros((2, df.get_hop_size()));

    // Process frames
    let start = Instant::now();
    let output = df.process_frames(input)?;
    let duration = start.elapsed();
    println!("Processed frame shape: {:?}", output.shape());
    println!("Time taken: {:?}", duration);

    Ok(())
}
