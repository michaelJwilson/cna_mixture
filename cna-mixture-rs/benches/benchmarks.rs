use cna_mixture_rs::{betabinom, betabinom_reduce, nbinom, nbinom_reduce, CnaEmission};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;

fn benchmark_cna_emission(c: &mut Criterion) {
    let ks: Vec<f64> = (1_000..=2_000).map(|x| x as f64).collect();
    let xs: Vec<f64> = ks.clone().into_iter().map(|x| 2. * x).collect();

    let bs: Vec<f64> = (10..=1_000).map(|x| x as f64).collect();
    let ns: Vec<f64> = bs.clone().into_iter().map(|x| 10. * x).collect();

    let means: Vec<f64> = (10..=20).map(|x| (x as f64)).collect();
    let weights = Array2::from_elem((ks.len(), means.len()), 1.0);

    c.bench_function("nbinom_reduce", |b| {
        b.iter(|| {
            nbinom_reduce(
                black_box(&ks),
                black_box(&xs),
                black_box(&means),
                black_box(0.01),
                black_box(weights.view()),
            );
        })
    });

    let alphas: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let betas: Vec<f64> = (1..=10).map(|x| x as f64).collect();

    c.bench_function("betabinom_reduce", |b| {
        b.iter(|| {
            betabinom_reduce(
                black_box(&bs),
                black_box(&ns),
                black_box(&alphas),
                black_box(&betas),
                black_box(weights.view()),
            );
        })
    });
}

fn benchmark_CnaEmission(c: &mut Criterion) {
    let ks: Vec<f64> = (1_000..=2_000).map(|x| x as f64).collect();
    let xs: Vec<f64> = ks.clone().into_iter().map(|x| 2. * x).collect();

    let bs: Vec<f64> = (10..=1_000).map(|x| x as f64).collect();
    let ns: Vec<f64> = bs.clone().into_iter().map(|x| 10. * x).collect();

    let means: Vec<f64> = (10..=20).map(|x| (x as f64)).collect();
    let weights = Array2::from_elem((ks.len(), means.len()), 1.0);

    let num_states = means.len();

    let mut cna_em = CnaEmission::new(num_states, ks, xs, bs, ns, true);
    cna_em.update_weights(weights.view());

    c.bench_function("nbinom", |b| {
        b.iter(|| {
            cna_em.nbinom(black_box(&means), black_box(0.01));
        })
    });

    c.bench_function("nbinom_reduce", |b| {
        b.iter(|| {
            cna_em.nbinom_reduce(black_box(&means), black_box(0.01));
        })
    });

    let alphas: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let betas: Vec<f64> = (1..=10).map(|x| x as f64).collect();

    c.bench_function("betabinom", |b| {
        b.iter(|| {
            let _result = cna_em.betabinom(black_box(&alphas), black_box(&betas));
        })
    });

    c.bench_function("betabinom_reduce", |b| {
        b.iter(|| {
            cna_em.betabinom_reduce(black_box(&alphas), black_box(&betas));
        })
    });
}

criterion_group!(benches, benchmark_CnaEmission);
//  criterion_group!(benches, benchmark_cna_emission);

criterion_main!(benches);
