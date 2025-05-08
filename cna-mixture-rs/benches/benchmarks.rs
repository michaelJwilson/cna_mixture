use cna_mixture_rs::nbinom_logpmf_core;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_cna_mixture_rs(c: &mut Criterion) {
    let k: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
    let r: Vec<f64> = (1..=1000).map(|x| (x as f64) * 0.5).collect();
    let p: Vec<f64> = vec![0.5; 1000]; // All elements set to 0.5

    c.bench_function("nbinom_logpmf_core", |b| {
        b.iter(|| {
            let _result = nbinom_logpmf_core(black_box(&k), black_box(&r), black_box(&p));
        })
    });
}

criterion_group!(benches, benchmark_cna_mixture_rs);
criterion_main!(benches);
