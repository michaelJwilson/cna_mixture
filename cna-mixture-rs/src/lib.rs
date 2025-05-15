extern crate statrs;

use itertools::izip;
use ndarray::parallel::prelude::IndexedParallelIterator;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::parallel::prelude::ParallelIterator;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use statrs::function::gamma::{digamma, ln_gamma};
use std::collections::HashMap;
use std::env;

pub struct CnaEmission {
    //  NB defining a struct associated locally in memory.
    ks: Vec<f64>,
    xs: Vec<f64>,
    bs: Vec<f64>,
    ns: Vec<f64>,
    nb_mapping: Vec<usize>,
    bb_mapping: Vec<usize>,
    thread_pool: ThreadPool,
}

impl CnaEmission {
    pub fn new(ks: Vec<f64>, xs: Vec<f64>, bs: Vec<f64>, ns: Vec<f64>) -> Self {
        let num_threads = env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(num_cpus::get());

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to build ThreadPool");

        let mut unique_nb_map: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), usize> =
            HashMap::new();

        let mut unique_bb_map: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), usize> =
            HashMap::new();

        let mut unique_ks: Vec<f64> = Vec::new();
        let mut unique_xs: Vec<f64> = Vec::new();

        let mut unique_bs: Vec<f64> = Vec::new();
        let mut unique_ns: Vec<f64> = Vec::new();

        let mut nb_mapping: Vec<usize> = Vec::new();
        let mut bb_mapping: Vec<usize> = Vec::new();

        for (&k, &x) in izip!(ks.iter(), xs.iter()) {
            let key = (OrderedFloat(k), OrderedFloat(x));

            if let Some(&index) = unique_nb_map.get(&key) {
                nb_mapping.push(index);
            } else {
                let new_index = unique_ks.len();

                unique_nb_map.insert(key, new_index);

                unique_ks.push(k);
                unique_xs.push(x);

                nb_mapping.push(new_index);
            }
        }

        for (&b, &n) in izip!(bs.iter(), ns.iter()) {
            let key = (OrderedFloat(b), OrderedFloat(n));

            if let Some(&index) = unique_bb_map.get(&key) {
                bb_mapping.push(index);
            } else {
                let new_index = unique_bs.len();

                unique_bb_map.insert(key, new_index);

                unique_bs.push(b);
                unique_ns.push(n);

                bb_mapping.push(new_index);
            }
        }

        CnaEmission {
            ks: unique_ks,
            xs: unique_xs,
            bs: unique_bs,
            ns: unique_ns,
            nb_mapping,
            bb_mapping,
            thread_pool,
        }
    }

    pub fn nbinom_logpmf_reduce(&self, means: &[f64], overdisp: f64) -> f64 {
        self.thread_pool
            .install(|| nbinom_logpmf_reduce(&self.ks, &self.xs, means, overdisp))
    }

    pub fn betabinom_logpmf_reduce(&self, alphas: &[f64], betas: &[f64]) -> f64 {
        self.thread_pool
            .install(|| betabinom_logpmf_reduce(&self.bs, &self.ns, alphas, betas))
    }
}

/*
#[pyclass]
struct CnaEmissionRs {
    inner: CnaEmission,
}

#[pymethods]
impl CnaEmissionRs {
    #[new]
    fn new(
        ks: PyReadonlyArray1<'_, f64>,
        xs: PyReadonlyArray1<'_, f64>,
        bs: PyReadonlyArray1<'_, f64>,
        ns: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let ks = ks.as_array().to_vec();
        let xs = xs.as_array().to_vec();
        let bs = bs.as_array().to_vec();
        let ns = ns.as_array().to_vec();

        let inner = CnaEmissionRs::new(ks, xs, bs, ns);

        Ok(CnaEmissionRs { inner })

    }

    fn nbinom_logpmf_reduce(&self, _means: PyReadonlyArray1<'_, f64>, overdisp: f64) -> f64 {
        let means = _means.as_array().to_vec();

        self.inner.nbinom_logpmf_reduce(&self.inner.ks, &self.inner.xs, &means, overdisp)
    }

    fn betabinom_logpmf_reduce(
        &self,
        _alphas: PyReadonlyArray1<'_, f64>,
        _betas: PyReadonlyArray1<'_, f64>,
    ) -> f64 {
        let alphas = _alphas.as_array().to_vec();
        let betas = _betas.as_array().to_vec();

        self.inner.betabinom_logpmf_reduce(&self.inner.bs, &self.inner.ns, &alphas, &betas)
    }
}
*/

//  NB  104.98 µs -> 70 µs (for all cores)
pub fn nbinom_logpmf_reduce(k: &[f64], x: &[f64], means: &[f64], overdisp: f64) -> f64 {
    let rr = 1.0 / overdisp;

    let result: f64 = k
        .par_iter()
        .zip(x.par_iter())
        .map(|(k_val, &x_val)| {
            let zero_point = -ln_gamma(1.0 + k_val);

            means
                .iter()
                .map(|&mean_val| {
                    let factor = 1.0 + overdisp * x_val * mean_val;
                    let ln_pp = -factor.ln();
                    let ln_qq = (1.0 - 1.0 / factor).ln();

                    let mut interim = zero_point;
                    interim += k_val * ln_qq + rr * ln_pp - ln_gamma(rr);
                    interim += ln_gamma(k_val + rr);

                    interim
                })
                .sum::<f64>()
        })
        .sum();

    result
}

//  NB  264.86 µs -> 91.962 µs
pub fn nbinom_logpmf(k: &[f64], x: &[f64], means: &[f64], overdisp: f64) -> Vec<Vec<f64>> {
    let rr = 1.0 / overdisp;

    let result: Vec<Vec<f64>> = k
        .par_iter()
        .zip(x.par_iter())
        .map(|(&k_val, &x_val)| {
            let zero_point = -ln_gamma(1.0 + k_val);

            let row: Vec<f64> = means
                .iter()
                .map(|&mean_val| {
                    let factor = 1.0 + overdisp * x_val * mean_val;
                    let ln_pp = -factor.ln();
                    let ln_qq = (1.0 - 1.0 / factor).ln();

                    let mut interim = zero_point;
                    interim += k_val * ln_qq + rr * ln_pp - ln_gamma(rr);
                    interim += ln_gamma(k_val + rr);

                    interim
                })
                .collect();

            row
        })
        .collect();

    result
}

//  NB  300 µs -> 108.68 µs (all cores)
pub fn betabinom_logpmf_reduce(k: &[f64], n: &[f64], a: &[f64], b: &[f64]) -> f64 {
    //
    //  Efficient beta binomial evaluation for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Beta-binomial_distribution
    let ga: Vec<f64> = a.iter().map(|&x| ln_gamma(x)).collect();
    let gb: Vec<f64> = b.iter().map(|&x| ln_gamma(x)).collect();
    let gab: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| ln_gamma(x + y))
        .collect();

    let result: f64 = k
        .par_iter()
        .zip(n.par_iter())
        .map(|(&k_val, &n_val)| {
            let zero_point =
                ln_gamma(n_val + 1.0) - ln_gamma(k_val + 1.0) - ln_gamma(n_val - k_val + 1.0);

            let sum: f64 = izip!(a, b, &ga, &gb, &gab)
                .map(|(&a_val, &b_val, &ga_val, &gb_val, &gab_val)| {
                    let mut interim = zero_point + gab_val - ga_val - gb_val;

                    interim += ln_gamma(k_val + a_val) + ln_gamma(n_val - k_val + b_val)
                        - ln_gamma(n_val + a_val + b_val);

                    interim
                })
                .sum();

            sum
        })
        .sum();

    result
}

//  NB  125.29 µs
pub fn betabinom_logpmf(k: &[f64], n: &[f64], a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    //
    //  Efficient beta binomial evaluation for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Beta-binomial_distribution
    let ga: Vec<f64> = a.iter().map(|&x| ln_gamma(x)).collect();
    let gb: Vec<f64> = b.iter().map(|&x| ln_gamma(x)).collect();
    let gab: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| ln_gamma(x + y))
        .collect();

    let result: Vec<Vec<f64>> = k
        .par_iter()
        .enumerate()
        .map(|(ii, &k_val)| {
            let zero_point =
                ln_gamma(n[ii] + 1.0) - ln_gamma(k_val + 1.0) - ln_gamma(n[ii] - k_val + 1.0);

            let row: Vec<f64> = a
                .iter()
                .enumerate()
                .map(|(ss, &a_val)| {
                    let mut interim = zero_point;

                    interim += ln_gamma(k_val + a_val) + ln_gamma(n[ii] - k_val + b[ss])
                        - ln_gamma(n[ii] + a_val + b[ss]);
                    interim += gab[ss] - ga[ss] - gb[ss];

                    interim
                })
                .collect();

            row
        })
        .collect::<Vec<Vec<f64>>>();

    result
}

#[pyfunction]
fn nbinom_logpmf_rs<'py>(
    k: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray1<'_, f64>,
    means: PyReadonlyArray1<'_, f64>,
    overdisp: f64,
) -> PyResult<f64> {
    // PyResult<Vec<Vec<f64>>>
    //
    //  Efficient negative binomial evaluation for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Negative_binomial_distribution
    let k = k.as_slice()?;
    let x = x.as_slice()?;

    let means = means.as_slice()?;

    Ok(nbinom_logpmf_reduce(&k, &x, &means, overdisp))
}

#[pyfunction]
fn betabinom_logpmf_rs<'py>(
    k: PyReadonlyArray1<'_, f64>,
    n: PyReadonlyArray1<'_, f64>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    //
    //  Efficient beta binomial evaluation for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Beta-binomial_distribution
    let k = k.to_vec()?;
    let n = n.to_vec()?;
    let a = a.to_vec()?;
    let b = b.to_vec()?;

    let result = betabinom_logpmf_reduce(&k, &n, &a, &b);

    Ok(result)
}

#[pyfunction]
fn grad_cna_mixture_em_cost_nb_rs<'py>(
    ks: PyReadonlyArray1<'_, f64>,
    mus: PyReadonlyArray1<'_, f64>,
    rs: PyReadonlyArray1<'_, f64>,
    phi: f64,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    //
    //  Gradient of the negative binomial component to the
    //  EM cost for the CNA mixture problem.
    //
    let ks = ks.to_vec()?;
    let mus = mus.to_vec()?;
    let rs = rs.to_vec()?;

    let zero_points: Vec<f64> = mus
        .iter()
        .zip(rs.iter())
        .map(|(&mu, &rr)| {
            digamma(rr) / (phi * phi) + (1.0 + phi * mu).ln() / phi / phi
                - phi * mu * rr / phi / (1.0 + phi * mu)
        })
        .collect();

    let result: (Vec<Vec<f64>>, Vec<Vec<f64>>) = ks
        .par_iter()
        .map(|&k_val| {
            let (mus_row, phi_row): (Vec<f64>, Vec<f64>) = mus
                .iter()
                .enumerate()
                .map(|(ss, &mu)| {
                    let mus_val = (k_val - phi * mu * rs[ss]) / mu / (1.0 + phi * mu);
                    let phi_val = zero_points[ss] - digamma(k_val + rs[ss]) / (phi * phi)
                        + k_val / phi / (1.0 + phi * mu);

                    (mus_val, phi_val)
                })
                .unzip();

            (mus_row, phi_row)
        })
        .unzip();

    Ok(result)
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
    //
    // 	Gradient of the	beta binomial component to the
    //  EM cost for the CNA mixture problem.
    //
    let ks = ks.to_vec()?;
    let ns = ns.to_vec()?;
    let alphas = alphas.to_vec()?;
    let betas = betas.to_vec()?;

    let zero_points: Vec<Vec<f64>> = alphas
        .iter()
        .zip(betas.iter())
        .map(|(&aa, &bb)| grad_ln_bb_ab_zeropoint(bb, aa))
        .collect();

    let result: (Vec<Vec<f64>>, Vec<Vec<f64>>) = ks
        .par_iter()
        .enumerate()
        .map(|(ii, &k_val)| {
            let (ps_row, tau_row): (Vec<f64>, Vec<f64>) = alphas
                .iter()
                .enumerate()
                .map(|(ss, &aa)| {
                    let tau = aa + betas[ss];
                    let baf = betas[ss] / tau;

                    let data_points = grad_ln_bb_ab_data(k_val, ns[ii], betas[ss], aa);
                    let interim = vector_sum(zero_points[ss].clone(), data_points);

                    let ps_val = -tau * interim[1] + tau * interim[0];
                    let tau_val = (1.0 - baf) * interim[1] + baf * interim[0];

                    (ps_val, tau_val)
                })
                .unzip();

            (ps_row, tau_row)
        })
        .unzip();

    Ok(result)
}

fn logsumexp(array: &[f64]) -> f64 {
    let max_val = array.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = array.iter().map(|&x| (x - max_val).exp()).sum();

    max_val + sum_exp.ln()
}

#[pyfunction]
fn ln_transition_probs_rs<'py>(
    num_states: usize,
    ln_fs: PyReadonlyArray2<'_, f64>,
    ln_bs: PyReadonlyArray2<'_, f64>,
    ln_trans: PyReadonlyArray2<'_, f64>,
    ln_ems: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<Vec<f64>>> {
    //
    //
    //
    //
    let ln_fs = ln_fs.as_array();
    let ln_bs = ln_bs.as_array();
    let ln_trans = ln_trans.as_array();
    let ln_ems = ln_ems.as_array();

    // TODO array for vectorization.
    let num_segments = ln_fs.shape()[0];
    let mut result: Vec<Vec<f64>> = vec![vec![0.0; num_states]; num_states];

    for ii in 0..(num_segments - 1) {
        for kk in 0..num_states {
            for ll in 0..num_states {
                result[kk][ll] += ln_trans[[kk, ll]]
                    + ln_ems[[ii + 1, ll]]
                    + ln_fs[[ii, kk]]
                    + ln_bs[[ii + 1, ll]];
            }
        }
    }

    //  TODO iter.
    for ii in 0..num_states {
        let norm = logsumexp(&result[ii]);

        for jj in 0..num_states {
            result[ii][jj] -= norm;
        }
    }

    Ok(result)
}

#[pymodule]
#[pyo3(name = "core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    //  m.add_class::<CnaEmissionRs>()?;
    m.add_function(wrap_pyfunction!(nbinom_logpmf_rs, m)?)?;
    m.add_function(wrap_pyfunction!(betabinom_logpmf_rs, m)?)?;
    m.add_function(wrap_pyfunction!(grad_cna_mixture_em_cost_nb_rs, m)?)?;
    m.add_function(wrap_pyfunction!(grad_cna_mixture_em_cost_bb_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ln_transition_probs_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp() {
        let array = vec![1.0, 2.0, 3.0];

        let result = logsumexp(&array);
        let expected = 3.4076059644443806;

        assert!(
            (result - expected).abs() < 1e-6,
            "result: {}, expected: {}",
            result,
            expected
        );
    }
}
