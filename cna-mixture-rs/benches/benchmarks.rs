use cna_mixture_rs::{
    betabinom_logpmf, betabinom_logpmf_reduce, nbinom_logpmf, nbinom_logpmf_reduce,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_cna_mixture_rs(c: &mut Criterion) {
    let k: Vec<f64> = (1_000..=2_000).map(|x| x as f64).collect();
    let x: Vec<f64> = k.clone().into_iter().map(|x| 2. * x).collect();

    let means: Vec<f64> = (10..=20).map(|x| (x as f64)).collect();

    c.bench_function("nbinom_logpmf", |b| {
        b.iter(|| {
            let _result = nbinom_logpmf_reduce(
                black_box(&k),
                black_box(&x),
                black_box(&means),
                black_box(0.01),
            );
        })
    });

    let bs: Vec<f64> = (10..=1_000).map(|x| x as f64).collect();
    let ns: Vec<f64> = k.clone().into_iter().map(|x| 10. * x).collect();

    let alphas: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let betas: Vec<f64> = (1..=10).map(|x| x as f64).collect();

    c.bench_function("betabinomial_logpmf_reduce", |b| {
        b.iter(|| {
            let _result = betabinom_logpmf_reduce(
                black_box(&bs),
                black_box(&ns),
                black_box(&alphas),
                black_box(&betas),
            );
        })
    });
}

criterion_group!(benches, benchmark_cna_mixture_rs);
criterion_main!(benches);
