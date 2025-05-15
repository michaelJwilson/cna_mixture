use cna_mixture_rs::{betabinom, betabinom_reduce, nbinom, nbinom_reduce, CnaEmission};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_cna_mixture_rs(c: &mut Criterion) {
    let ks: Vec<f64> = (1_000..=2_000).map(|x| x as f64).collect();
    let xs: Vec<f64> = ks.clone().into_iter().map(|x| 2. * x).collect();

    let bs: Vec<f64> = (10..=1_000).map(|x| x as f64).collect();
    let ns: Vec<f64> = bs.clone().into_iter().map(|x| 10. * x).collect();

    let cna_em = CnaEmission::new(ks, xs, bs, ns, true);

    let means: Vec<f64> = (10..=20).map(|x| (x as f64)).collect();

    c.bench_function("nbinom", |b| {
        b.iter(|| {
            let _result = cna_em.nbinom(black_box(&means), black_box(0.01));
        })
    });

    c.bench_function("nbinom_reduce", |b| {
        b.iter(|| {
            let _result = cna_em.nbinom_reduce(black_box(&means), black_box(0.01));
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
            let _result = cna_em.betabinom_reduce(black_box(&alphas), black_box(&betas));
        })
    });
}

criterion_group!(benches, benchmark_cna_mixture_rs);
criterion_main!(benches);
