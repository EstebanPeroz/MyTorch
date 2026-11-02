use std::path::{Path, PathBuf};

use clap::{Arg, ArgMatches, Command};
use cli_core::Cli;

pub struct NetworkRequest {
    config_file: PathBuf,
    count: usize,
}

impl NetworkRequest {
    pub fn config_file(&self) -> &Path {
        &self.config_file
    }

    pub fn count(&self) -> usize {
        self.count
    }

    fn from_pair(config_file: &str, count: &str) -> Result<Self, String> {
        let count = count
            .parse::<usize>()
            .ok()
            .filter(|&count| count > 0)
            .ok_or_else(|| format!("error: invalid number of networks: '{count}'"))?;
        Ok(NetworkRequest {
            config_file: PathBuf::from(config_file),
            count,
        })
    }
}

pub struct Args {
    requests: Vec<NetworkRequest>,
}

impl Args {
    pub fn requests(&self) -> &[NetworkRequest] {
        &self.requests
    }
}

impl Cli for Args {
    const HELP: &'static str = include_str!("help.txt");

    fn command() -> Command {
        Command::new("my_torch_generator").arg(
            Arg::new("PAIRS")
                .value_name("config_file nb")
                .num_args(2..)
                .required(true),
        )
    }

    fn from_matches(matches: &ArgMatches) -> Result<Self, String> {
        let values: Vec<&String> = matches.get_many::<String>("PAIRS").unwrap().collect();
        if !values.len().is_multiple_of(2) {
            return Err(
                "error: each configuration file must be followed by a number of networks".into(),
            );
        }
        let requests = values
            .chunks(2)
            .map(|pair| NetworkRequest::from_pair(pair[0], pair[1]))
            .collect::<Result<_, _>>()?;
        Ok(Args { requests })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Args>, String> {
        Args::try_parse_from(std::iter::once("my_torch_generator").chain(args.iter().copied()))
    }

    #[test]
    fn parses_several_pairs() {
        let args = parse(&["basic.conf", "3", "deep.conf", "1"])
            .unwrap()
            .unwrap();
        let requests: Vec<_> = args
            .requests()
            .iter()
            .map(|request| (request.config_file().to_str().unwrap(), request.count()))
            .collect();
        assert_eq!(requests, [("basic.conf", 3), ("deep.conf", 1)]);
    }
}
