use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct Rng {
    generator: StdRng,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng {
            generator: StdRng::seed_from_u64(seed),
        }
    }

    pub(crate) fn generator(&mut self) -> &mut StdRng {
        &mut self.generator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    fn draws(seed: u64) -> Vec<u64> {
        let mut rng = Rng::new(seed);
        (0..5).map(|_| rng.generator().random()).collect()
    }

    #[test]
    fn the_same_seed_gives_the_same_numbers() {
        assert_eq!(draws(7), draws(7));
    }

    #[test]
    fn another_seed_gives_other_numbers() {
        assert_ne!(draws(7), draws(8));
    }
}
