//! The binary format of network files. See docs/my_torch.md for the layout.

use std::collections::BTreeMap;

use crate::activations::Activation;
use crate::dense::Dense;
use crate::matrix::{Matrix, Scalar};
use crate::model::Model;
use crate::network::Network;
use crate::trainer::Trainer;
use crate::training_file::TrainingFile;

const MAGIC: [u8; 4] = *b"MTNN";
pub(crate) const FORMAT_VERSION: u32 = 1;
const SCALAR_SIZE: usize = size_of::<Scalar>();

pub(crate) fn encode(layers: &[Dense], trainer: &Trainer, seed: u64) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    put_u32(&mut out, FORMAT_VERSION);
    put_count(&mut out, layers.len())?;
    for (index, layer) in layers.iter().enumerate() {
        encode_layer(&mut out, layer).map_err(|error| format!("layer {}: {error}", index + 1))?;
    }
    put_str(&mut out, trainer.loss().name())?;
    put_str(&mut out, trainer.optimizer().name())?;
    let settings = trainer.optimizer().settings();
    put_count(&mut out, settings.len())?;
    for (name, value) in &settings {
        put_str(&mut out, name)?;
        put_scalar(&mut out, *value);
    }
    put_u64(&mut out, trainer.batch_size() as u64);
    put_u64(&mut out, trainer.epochs() as u64);
    put_u64(&mut out, seed);
    match trainer.early_stopping() {
        Some(early_stopping) => {
            out.push(1);
            put_scalar(&mut out, early_stopping.validation());
            put_u64(&mut out, early_stopping.patience() as u64);
        }
        None => out.push(0),
    }
    Ok(out)
}

fn encode_layer(out: &mut Vec<u8>, layer: &Dense) -> Result<(), String> {
    let parameters = layer.weights().data().iter().chain(layer.biases().data());
    if parameters.clone().any(|value| !value.is_finite()) {
        return Err(
            "a weight or bias is not a finite number, so the network cannot be saved".into(),
        );
    }
    put_str(out, layer.activation().name())?;
    put_count(out, layer.input_size())?;
    put_count(out, layer.output_size())?;
    for value in parameters {
        put_scalar(out, *value);
    }
    Ok(())
}

pub(crate) fn decode(bytes: &[u8]) -> Result<Model, String> {
    let mut reader = Reader { bytes, position: 0 };
    let magic = reader.take(MAGIC.len(), "the header")?;
    if magic != MAGIC {
        return Err("not a network file (wrong header)".into());
    }
    let version = reader.u32("the version")?;
    if version != FORMAT_VERSION {
        return Err(format!(
            "unsupported network file version {version}, expected {FORMAT_VERSION}"
        ));
    }
    let count = reader.u32("the number of layers")? as usize;
    let mut layers = Vec::new();
    for index in 0..count {
        let layer =
            decode_layer(&mut reader).map_err(|error| format!("layer {}: {error}", index + 1))?;
        layers.push(layer);
    }
    let loss = reader.string("the loss")?;
    let optimizer = reader.string("the optimizer")?;
    let count = reader.u32("the number of optimizer settings")? as usize;
    let mut settings = BTreeMap::new();
    for _ in 0..count {
        let name = reader.string("an optimizer setting")?;
        settings.insert(name, reader.scalar("an optimizer setting")?);
    }
    let batch_size = reader.usize("the batch size")?;
    let epochs = reader.usize("the number of epochs")?;
    let seed = reader.u64("the seed")?;
    let early_stopping = match reader.take(1, "the early stopping flag")?[0] {
        0 => None,
        1 => Some((
            reader.scalar("the validation share")?,
            reader.usize("the patience")?,
        )),
        flag => {
            return Err(format!(
                "invalid early stopping flag {flag}, expected 0 or 1"
            ));
        }
    };
    if reader.position != bytes.len() {
        return Err(format!(
            "{} unexpected bytes after the end of the network",
            bytes.len() - reader.position
        ));
    }
    let training = TrainingFile::from_parts(
        loss,
        optimizer,
        settings,
        batch_size,
        epochs,
        seed,
        early_stopping,
    );
    Model::new(Network::new(layers)?, training.trainer()?, seed)
}

fn decode_layer(reader: &mut Reader) -> Result<Dense, String> {
    let activation = Activation::from_name(&reader.string("an activation")?)?;
    let inputs = reader.u32("the number of inputs")? as usize;
    let outputs = reader.u32("the number of neurons")? as usize;
    let weights = reader.scalars(inputs, outputs, "the weights")?;
    let biases = reader.scalars(1, outputs, "the biases")?;
    let layer = Dense::new(
        Matrix::from_vec(inputs, outputs, weights)?,
        Matrix::from_vec(1, outputs, biases)?,
        activation,
    )?;
    let parameters = layer.weights().data().iter().chain(layer.biases().data());
    if parameters.into_iter().any(|value| !value.is_finite()) {
        return Err("a weight or bias is not a finite number".into());
    }
    Ok(layer)
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_scalar(out: &mut Vec<u8>, value: Scalar) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_count(out: &mut Vec<u8>, count: usize) -> Result<(), String> {
    let count = u32::try_from(count).map_err(|_| format!("{count} is too large to be saved"))?;
    put_u32(out, count);
    Ok(())
}

/// A name: one byte of length, then its UTF-8 bytes.
fn put_str(out: &mut Vec<u8>, text: &str) -> Result<(), String> {
    let length =
        u8::try_from(text.len()).map_err(|_| format!("the name \"{text}\" is too long"))?;
    out.push(length);
    out.extend_from_slice(text.as_bytes());
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize, what: &str) -> Result<&[u8], String> {
        let end = self
            .position
            .checked_add(count)
            .filter(|&end| end <= self.bytes.len());
        let Some(end) = end else {
            return Err(format!(
                "the file ends too early while reading {what} ({count} bytes needed, {} left)",
                self.bytes.len() - self.position
            ));
        };
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn u32(&mut self, what: &str) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }

    fn u64(&mut self, what: &str) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }

    fn usize(&mut self, what: &str) -> Result<usize, String> {
        let value = self.u64(what)?;
        usize::try_from(value).map_err(|_| format!("{what} is too large: {value}"))
    }

    fn scalar(&mut self, what: &str) -> Result<Scalar, String> {
        Ok(Scalar::from_le_bytes(
            self.take(SCALAR_SIZE, what)?.try_into().unwrap(),
        ))
    }

    fn string(&mut self, what: &str) -> Result<String, String> {
        let length = usize::from(self.take(1, what)?[0]);
        let bytes = self.take(length, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| format!("{what} is not valid text"))
    }

    /// `rows × cols` numbers. The size is checked against what is left before anything is allocated,
    /// so a corrupted size cannot ask for gigabytes.
    fn scalars(&mut self, rows: usize, cols: usize, what: &str) -> Result<Vec<Scalar>, String> {
        let count = rows
            .checked_mul(cols)
            .and_then(|count| count.checked_mul(SCALAR_SIZE))
            .ok_or_else(|| format!("{what} are too many ({rows} × {cols})"))?;
        let bytes = self.take(count, what)?;
        let (chunks, _) = bytes.as_chunks::<SCALAR_SIZE>();
        Ok(chunks
            .iter()
            .map(|chunk| Scalar::from_le_bytes(*chunk))
            .collect())
    }
}
