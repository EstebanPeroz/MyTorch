use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Duration;

use my_torch::{Network, Scalar};
use serde_json::{Value, json};

use crate::confusion::Confusion;
use crate::label::Label;

const PAGE: &str = include_str!("dashboard.html");
const REQUEST_LIMIT: usize = 8192;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a finished dashboard stays up, so that an open page, which asks for news twice a second, can
/// learn that the training is over before the program exits.
const LINGER: Duration = Duration::from_millis(1500);

/// The measures of one epoch. A value that is not a number is shown as a gap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub epoch: usize,
    pub training_loss: Scalar,
    pub validation_loss: Option<Scalar>,
    pub training_accuracy: Scalar,
    pub validation_accuracy: Option<Scalar>,
}

impl Point {
    fn to_json(self) -> Value {
        json!({
            "epoch": self.epoch,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
        })
    }
}

/// The samples a confusion matrix was measured on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Set {
    Training,
    Validation,
}

impl Set {
    fn name(self) -> &'static str {
        match self {
            Set::Training => "training",
            Set::Validation => "validation",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Status {
    Running,
    Done,
    Failed(String),
}

/// One dense layer of the network, as the page draws it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layer {
    inputs: usize,
    outputs: usize,
    activation: &'static str,
}

impl Layer {
    /// The weights plus the biases.
    fn parameters(self) -> usize {
        self.inputs * self.outputs + self.outputs
    }
}

#[derive(Debug)]
struct State {
    epochs: usize,
    layers: Vec<Layer>,
    points: Vec<Point>,
    confusion: Option<(Set, Confusion)>,
    status: Status,
}

impl State {
    fn to_json(&self) -> Value {
        let (status, message) = match &self.status {
            Status::Running => ("running", ""),
            Status::Done => ("done", ""),
            Status::Failed(message) => ("failed", message.as_str()),
        };
        json!({
            "epochs": self.epochs,
            "layers": self.layers.iter().map(|layer| json!({
                "inputs": layer.inputs,
                "outputs": layer.outputs,
                "activation": layer.activation,
                "parameters": layer.parameters(),
            })).collect::<Vec<_>>(),
            "status": status,
            "message": message,
            "points": self.points.iter().map(|point| point.to_json()).collect::<Vec<_>>(),
            "confusion": self.confusion.as_ref().map(|(set, confusion)| json!({
                "on": set.name(),
                "labels": Label::ALL.map(|label| label.to_string()),
                "counts": confusion.counts(),
            })),
        })
    }
}

type Shared = Arc<Mutex<State>>;

fn lock(state: &Shared) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A live view of a training, served as a web page on this machine only (127.0.0.1).
///
/// The page asks `/metrics` twice a second and draws the curves itself, so it needs no internet access. It
/// keeps what it has received: after the program exits, the page still shows the curves, marked as done.
pub struct Dashboard {
    state: Shared,
    address: SocketAddr,
}

impl Dashboard {
    /// Starts serving on a free port. `epochs` is the most epochs the training may run, and `network` is the
    /// shape the page shows.
    pub fn start(epochs: usize, network: &Network) -> Result<Dashboard, String> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .map_err(|error| format!("cannot start the dashboard: {error}"))?;
        let address = listener
            .local_addr()
            .map_err(|error| format!("cannot start the dashboard: {error}"))?;
        let state = Arc::new(Mutex::new(State {
            epochs,
            layers: network
                .layers()
                .iter()
                .map(|layer| Layer {
                    inputs: layer.input_size(),
                    outputs: layer.output_size(),
                    activation: layer.activation().name(),
                })
                .collect(),
            points: Vec::new(),
            confusion: None,
            status: Status::Running,
        }));
        let served = Arc::clone(&state);
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let served = Arc::clone(&served);
                thread::spawn(move || {
                    let _ = serve(stream, &served);
                });
            }
        });
        Ok(Dashboard { state, address })
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    /// Adds an epoch. `confusion` replaces the matrix the page shows: the labels the network gives to the
    /// samples of one set after this epoch.
    pub fn record(&self, point: Point, confusion: Option<(Set, Confusion)>) {
        let mut state = lock(&self.state);
        state.points.push(point);
        if confusion.is_some() {
            state.confusion = confusion;
        }
    }

    /// Tells the page how the training ended, then keeps serving for a moment.
    pub fn finish(self, result: Result<(), &str>) {
        self.close(result, LINGER);
    }

    fn close(self, result: Result<(), &str>, linger: Duration) {
        lock(&self.state).status = match result {
            Ok(()) => Status::Done,
            Err(message) => Status::Failed(message.to_string()),
        };
        thread::sleep(linger);
    }
}

fn serve(mut stream: TcpStream, state: &Shared) -> std::io::Result<()> {
    stream.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    stream.set_write_timeout(Some(REQUEST_TIMEOUT))?;
    let request = read_request(&mut stream)?;
    let mut words = request.lines().next().unwrap_or("").split_whitespace();
    let (method, target) = (words.next(), words.next());
    let path = target.map(|target| target.split(['?', '#']).next().unwrap_or(""));
    let (status, kind, body) = match (method, path) {
        (Some("GET"), Some("/")) => ("200 OK", "text/html; charset=utf-8", PAGE.to_string()),
        (Some("GET"), Some("/metrics")) => (
            "200 OK",
            "application/json",
            lock(state).to_json().to_string(),
        ),
        (Some("GET"), _) => ("404 Not Found", "text/plain", "not found".to_string()),
        _ => (
            "405 Method Not Allowed",
            "text/plain",
            "method not allowed".to_string(),
        ),
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<String> {
    let mut request = Vec::new();
    let mut buffer = [0; 1024];
    while !request.windows(4).any(|window| window == b"\r\n\r\n") && request.len() < REQUEST_LIMIT {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..read]);
    }
    Ok(String::from_utf8_lossy(&request).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use my_torch::{Activation, Dense, Matrix};

    fn network() -> Network {
        let layer = |inputs, outputs, activation| {
            Dense::new(
                Matrix::zeros(inputs, outputs),
                Matrix::zeros(1, outputs),
                activation,
            )
            .unwrap()
        };
        Network::new(vec![
            layer(4, 3, Activation::Relu),
            layer(3, 2, Activation::Identity),
        ])
        .unwrap()
    }

    fn start(epochs: usize) -> Dashboard {
        Dashboard::start(epochs, &network()).unwrap()
    }

    fn get(dashboard: &Dashboard, target: &str) -> (String, String) {
        request(
            dashboard,
            &format!("GET {target} HTTP/1.1\r\nHost: x\r\n\r\n"),
        )
    }

    fn request(dashboard: &Dashboard, text: &str) -> (String, String) {
        let mut stream = TcpStream::connect(dashboard.address).unwrap();
        stream.write_all(text.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        (head.lines().next().unwrap().to_string(), body.to_string())
    }

    fn point(epoch: usize) -> Point {
        Point {
            epoch,
            training_loss: 1.0 / epoch as Scalar,
            validation_loss: Some(2.0),
            training_accuracy: 0.5,
            validation_accuracy: None,
        }
    }

    fn metrics(dashboard: &Dashboard) -> Value {
        serde_json::from_str(&get(dashboard, "/metrics").1).unwrap()
    }

    #[test]
    fn listens_on_this_machine_only() {
        let dashboard = start(10);
        assert!(dashboard.url().starts_with("http://127.0.0.1:"));
    }

    #[test]
    fn serves_the_page() {
        let dashboard = start(10);
        let (status, body) = get(&dashboard, "/");
        assert_eq!(status, "HTTP/1.1 200 OK");
        assert_eq!(body, PAGE);
        assert_eq!(get(&dashboard, "/?refresh=1").1, PAGE);
    }

    #[test]
    fn serves_the_epochs_recorded_so_far() {
        let dashboard = start(10);
        let first = metrics(&dashboard);
        assert_eq!(first["epochs"], 10);
        assert_eq!(first["status"], "running");
        assert_eq!(first["points"], json!([]));

        dashboard.record(point(1), None);
        dashboard.record(point(2), None);
        let later = metrics(&dashboard);
        assert_eq!(later["points"].as_array().unwrap().len(), 2);
        assert_eq!(later["points"][1]["epoch"], 2);
        assert_eq!(later["points"][1]["training_loss"], 0.5);
        assert_eq!(later["points"][1]["validation_loss"], 2.0);
        assert_eq!(later["points"][1]["validation_accuracy"], Value::Null);
    }

    #[test]
    fn serves_the_shape_of_the_network() {
        let layers = &metrics(&start(1))["layers"];
        assert_eq!(layers.as_array().unwrap().len(), 2);
        assert_eq!(layers[0]["inputs"], 4);
        assert_eq!(layers[0]["outputs"], 3);
        assert_eq!(layers[0]["activation"], Activation::Relu.name());
        assert_eq!(layers[0]["parameters"], 15);
        assert_eq!(layers[1]["parameters"], 8);
    }

    #[test]
    fn serves_the_latest_confusion_matrix_with_the_label_names() {
        let dashboard = start(2);
        assert_eq!(metrics(&dashboard)["confusion"], Value::Null);

        let mut first = Confusion::default();
        first.record(Label::Nothing, Label::Stalemate);
        dashboard.record(point(1), Some((Set::Validation, first)));
        let mut second = Confusion::default();
        second.record(Label::Stalemate, Label::Stalemate);
        second.record(Label::Stalemate, Label::Stalemate);
        dashboard.record(point(2), Some((Set::Validation, second)));
        // An epoch without a matrix keeps the last one.
        dashboard.record(point(3), None);

        let confusion = &metrics(&dashboard)["confusion"];
        assert_eq!(confusion["on"], "validation");
        assert_eq!(confusion["labels"][0], "Nothing");
        assert_eq!(confusion["labels"][5], "Stalemate");
        assert_eq!(confusion["counts"][5][5], 2);
        assert_eq!(confusion["counts"][0][5], 0);
    }

    #[test]
    fn a_value_that_is_not_a_number_becomes_null() {
        let dashboard = start(1);
        dashboard.record(
            Point {
                training_loss: Scalar::NAN,
                ..point(1)
            },
            None,
        );
        assert_eq!(
            metrics(&dashboard)["points"][0]["training_loss"],
            Value::Null
        );
    }

    #[test]
    fn tells_how_the_training_ended() {
        let done = start(1);
        let state = Arc::clone(&done.state);
        done.close(Ok(()), Duration::ZERO);
        assert_eq!(lock(&state).status, Status::Done);

        let failed = start(1);
        let state = Arc::clone(&failed.state);
        failed.close(Err("diverged"), Duration::ZERO);
        let json = lock(&state).to_json();
        assert_eq!(json["status"], "failed");
        assert_eq!(json["message"], "diverged");
    }

    #[test]
    fn refuses_what_it_does_not_serve() {
        let dashboard = start(1);
        assert_eq!(get(&dashboard, "/missing").0, "HTTP/1.1 404 Not Found");
        let post = request(&dashboard, "POST /metrics HTTP/1.1\r\n\r\n");
        assert_eq!(post.0, "HTTP/1.1 405 Method Not Allowed");
        assert_eq!(
            request(&dashboard, "garbage\r\n\r\n").0,
            "HTTP/1.1 405 Method Not Allowed"
        );
    }

    #[test]
    fn a_silent_client_does_not_block_the_others() {
        let dashboard = start(1);
        let _silent = TcpStream::connect(dashboard.address).unwrap();
        assert_eq!(get(&dashboard, "/metrics").0, "HTTP/1.1 200 OK");
    }
}
