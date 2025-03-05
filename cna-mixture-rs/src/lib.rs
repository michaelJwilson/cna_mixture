extern crate statrs;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use statrs::function::gamma::{ln_gamma, digamma};
use statrs::function::factorial::ln_factorial;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::parallel::prelude::IndexedParallelIterator;
use ndarray::parallel::prelude::ParallelIterator;
use rayon::ThreadPoolBuilder;
use once_cell::sync::Lazy;
use std::env;
use num_cpus;

static THREAD_POOL: Lazy<rayon::ThreadPool> = Lazy::new(|| {
    let num_threads = env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| num_cpus::get());
    ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap()
});

#[pyfunction]
fn nbinom_logpmf<'py>(
    k: PyReadonlyArray1<'_, f64>,
    r: PyReadonlyArray1<'_, f64>,
    p: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<Vec<f64>>> {
    //  see: https://en.wikipedia.org/wiki/Negative_binomial_distribution
    let k = k.to_vec()?;
    let r = r.to_vec()?;
    let p = p.to_vec()?;

    // NB parameter-dependent only
    let gr: Vec<f64> = r.iter().map(|&x| ln_gamma(x)).collect();
    let lnp: Vec<f64> = p.iter().map(|&x| x.ln()).collect();
    let lnq: Vec<f64> = p.iter().map(|&x| (1. - x).ln()).collect();

    let result: Vec<Vec<f64>> = THREAD_POOL.install(|| {
        k.par_iter().enumerate().map(|(_ii, &k_val)| {
	    //  NB data-dependent only
            let zero_point = -ln_factorial(k_val as u64);
	    
            let row: Vec<f64> = r.iter().enumerate().map(|(ss, &r_val)| {
                let mut interim = zero_point;

                interim += k_val * lnq[ss] + r_val * lnp[ss] - gr[ss] ;
                interim += ln_gamma(k_val + r_val);

		interim

            }).collect();

            row

            }).collect::<Vec<Vec<f64>>>()
    });

    Ok(result)
}

#[pyfunction]
fn betabinom_logpmf<'py>(
    k: PyReadonlyArray1<'_, f64>,
    n: PyReadonlyArray1<'_, f64>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<Vec<f64>>> {
    //  see: https://en.wikipedia.org/wiki/Beta-binomial_distribution
    let k = k.to_vec()?;
    let n = n.to_vec()?;
    let a = a.to_vec()?;
    let b = b.to_vec()?;

    let ga: Vec<f64> = a.iter().map(|&x| ln_gamma(x)).collect();
    let gb: Vec<f64> = b.iter().map(|&x| ln_gamma(x)).collect();
    let gab: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| ln_gamma(x + y)).collect();

    // let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    let result: Vec<Vec<f64>> = THREAD_POOL.install(|| {
        k.par_iter().enumerate().map(|(ii, &k_val)| {
            let zero_point = ln_gamma(n[ii] + 1.0) - ln_gamma(k_val + 1.0) - ln_gamma(n[ii] - k_val + 1.0);	
            let row: Vec<f64> = a.iter().enumerate().map(|(ss, &a_val)| {
            	let mut interim = zero_point;
	    
		interim += ln_gamma(k_val + a_val) + ln_gamma(n[ii] - k_val + b[ss]) - ln_gamma(n[ii] + a_val + b[ss]);
            	interim += gab[ss] - ga[ss] - gb[ss];
	    
		interim
	    
            }).collect();
	
	    row
	
    	    }).collect::<Vec<Vec<f64>>>()
    });

    Ok(result)
}

#[pyfunction]
fn grad_cna_mixture_em_cost_nb_rs<'py>(
    ks: PyReadonlyArray1<'_, f64>,
    mus: PyReadonlyArray1<'_, f64>,
    rs: PyReadonlyArray1<'_, f64>,
    phi: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let ks = ks.to_vec()?;
    let mus = mus.to_vec()?;
    let rs = rs.to_vec()?;

    let zero_points: Vec<f64> = mus.iter().zip(rs.iter()).map(|(&mu, &rr)| digamma(rr) / (phi * phi) + (1.0 + phi * mu).ln() / phi / phi - phi * mu * rr / phi / (1.0 + phi * mu)).collect();

    let mus_result: Vec<Vec<f64>> = THREAD_POOL.install(|| {
        ks.par_iter().enumerate().map(|(_ii, &k_val)| {
            let row: Vec<f64> = mus.iter().zip(rs.iter()).map(|(&mu, &rr)| {
                (k_val - phi * mu * rr) / mu / (1.0 + phi * mu)
            }).collect();

            row

            }).collect::<Vec<Vec<f64>>>()
    });

    let phi_result: Vec<Vec<f64>> = THREAD_POOL.install(|| {
        ks.par_iter().enumerate().map(|(_ii, &k_val)| {
            let row: Vec<f64> = mus.iter().enumerate().map(|(ss, &mu)| {
	    	zero_points[ss] - digamma(k_val + rs[ss]) / (phi * phi) + k_val / phi / (1.0 + phi * mu)
            }).collect();

            row

            }).collect::<Vec<Vec<f64>>>()
    });

    Ok((mus_result, phi_result))
}

fn vector_sum(vec1: Vec<f64>, vec2: Vec<f64>) -> Vec<f64> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a + b).collect()
}

fn grad_ln_bb_ab_zeropoint(a: f64, b: f64) -> Vec<f64> {
   let gab = digamma(a + b);
   let ga = digamma(a);
   let gb = digamma(b);

   vec![gab - ga, gab - gb]
}

fn grad_ln_bb_ab_data(k: f64, n: f64, a: f64, b: f64) -> Vec<f64> {
   let gka = digamma(k + a);
   let gnab = digamma(n + a + b);
   let gnkb = digamma(n - k + b);

   vec![gka - gnab, gnkb - gnab]
}

#[pyfunction]
fn grad_cna_mixture_em_cost_bb_rs<'py>(
    ks: PyReadonlyArray1<'_, f64>,
    ns: PyReadonlyArray1<'_, f64>,
    alphas: PyReadonlyArray1<'_, f64>,
    betas: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let ks = ks.to_vec()?;
    let ns = ns.to_vec()?;
    let alphas = alphas.to_vec()?;
    let betas = betas.to_vec()?;

    let zero_points: Vec<Vec<f64>> = alphas.iter().zip(betas.iter()).map(|(&aa, &bb)| {
    	grad_ln_bb_ab_zeropoint(bb, aa)
    }).collect();

    let ps_result: Vec<Vec<f64>> = THREAD_POOL.install(|| {
        ks.par_iter().enumerate().map(|(ii, &k_val)| {  
            let row: Vec<f64> = alphas.iter().enumerate().map(|(ss, &aa)| {
	    	let tau = aa + betas[ss];
		let data_points = grad_ln_bb_ab_data(k_val, ns[ii], betas[ss], aa);
	    	let interim = vector_sum(zero_points[ss].clone(), data_points); 

		-tau * interim[1] + tau * interim[0]
            }).collect();

            row

            }).collect::<Vec<Vec<f64>>>()
    });

    let tau_result: Vec<Vec<f64>> = THREAD_POOL.install(|| {
        ks.par_iter().enumerate().map(|(ii, &k_val)| {
            let row: Vec<f64> = alphas.iter().enumerate().map(|(ss, &aa)| {
            	let tau	= aa + betas[ss];
		let baf = betas[ss] / tau;

		let data_points	= grad_ln_bb_ab_data(k_val, ns[ii], betas[ss], aa);
		let interim = vector_sum(zero_points[ss].clone(), data_points);

		(1.0 - baf) * interim[1] + baf * interim[0]
            }).collect();

            row

            }).collect::<Vec<Vec<f64>>>()
    });

    Ok((ps_result, tau_result))
}

#[pymodule]
#[pyo3(name = "core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nbinom_logpmf, m)?)?;
    m.add_function(wrap_pyfunction!(betabinom_logpmf, m)?)?;
    m.add_function(wrap_pyfunction!(grad_cna_mixture_em_cost_nb_rs, m)?)?;
    m.add_function(wrap_pyfunction!(grad_cna_mixture_em_cost_bb_rs, m)?)?;
    Ok(())
}
