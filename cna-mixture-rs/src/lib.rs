extern crate statrs;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use statrs::function::gamma::ln_gamma;
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

#[pymodule]
#[pyo3(name = "core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nbinom_logpmf, m)?)?;
    m.add_function(wrap_pyfunction!(betabinom_logpmf, m)?)?;
    Ok(())
}
