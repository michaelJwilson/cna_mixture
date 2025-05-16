extern crate statrs;

use itertools::izip;
use ndarray::parallel::prelude::IndexedParallelIterator;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::parallel::prelude::ParallelIterator;
use ndarray::{Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::{ThreadPool, ThreadPoolBuilder};

use statrs::function::gamma::{digamma, ln_gamma};
use std::collections::HashMap;
use std::env;

pub struct CnaEmission {
    ks: Vec<f64>,
    xs: Vec<f64>,
    bs: Vec<f64>,
    ns: Vec<f64>,
    nb_weights: Array2<f64>,
    bb_weights: Array2<f64>,
    thread_pool: ThreadPool,
}

impl CnaEmission {
    pub fn new(
        ks: Vec<f64>,
        xs: Vec<f64>,
        bs: Vec<f64>,
        ns: Vec<f64>,
        weights: Array2<f64>,
        compress: bool,
    ) -> Self {
        let num_threads = env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| num_cpus::get());

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to build ThreadPool");

        if compress {
            let mut unique_nb_map: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), usize> =
                HashMap::new();

            let mut unique_bb_map: HashMap<(OrderedFloat<f64>, OrderedFloat<f64>), usize> =
                HashMap::new();

            let mut unique_ks: Vec<f64> = Vec::new();
            let mut unique_xs: Vec<f64> = Vec::new();

            let mut unique_bs: Vec<f64> = Vec::new();
            let mut unique_ns: Vec<f64> = Vec::new();

            let mut nb_weights = Array2::<f64>::zeros((0, weights.shape()[1]));
            let mut bb_weights = Array2::<f64>::zeros((0, weights.shape()[1]));

            for (&k, &x, weights_row) in izip!(ks.iter(), xs.iter(), weights.axis_iter(Axis(0)).into_iter()) {
                let key = (OrderedFloat(k), OrderedFloat(x));

                if let Some(&index) = unique_nb_map.get(&key) {
                    nb_weights
                        .row_mut(index)
                        .iter_mut()
                        .zip(weights_row.iter())
                        .for_each(|(w, &v)| *w += v);
                } else {
                    let new_index = unique_ks.len();

                    unique_nb_map.insert(key, new_index);

                    unique_ks.push(k);
                    unique_xs.push(x);

                    let new_row = weights_row.to_owned();

                    nb_weights.push_row(new_row.view()).unwrap();
                }
            }

            for (&b, &n, weights_row) in izip!(bs.iter(), ns.iter(), weights.axis_iter(Axis(0)).into_iter()) {
                let key = (OrderedFloat(b), OrderedFloat(n));

                if let Some(&index) = unique_bb_map.get(&key) {
                    bb_weights
                        .row_mut(index)
                        .iter_mut()
                        .zip(weights_row.iter())
                        .for_each(|(w, &v)| *w += v);
                } else {
                    let new_index = unique_bs.len();

                    unique_bb_map.insert(key, new_index);

                    unique_bs.push(b);
                    unique_ns.push(n);

                    let new_row = weights_row.to_owned();
                    bb_weights.push_row(new_row.view()).unwrap();
                }
            }

            CnaEmission {
                ks: unique_ks,
                xs: unique_xs,
                bs: unique_bs,
                ns: unique_ns,
                nb_weights,
                bb_weights,
                thread_pool,
            }
        } else {
            // In the uncompressed case, use the original weights
            CnaEmission {
                ks,
                xs,
                bs,
                ns,
                nb_weights: weights.clone(),
                bb_weights: weights,
                thread_pool,
            }
        }
    }

    pub fn nbinom(&self, means: &[f64], overdisp: f64) -> Vec<Vec<f64>> {
        self.thread_pool
            .install(|| nbinom(&self.ks, &self.xs, means, overdisp))
    }

    pub fn nbinom_reduce(&self, means: &[f64], overdisp: f64) -> f64 {
        self.thread_pool
            .install(|| nbinom_reduce(&self.ks, &self.xs, means, overdisp, self.nb_weights.view()))
    }

    pub fn betabinom(&self, alphas: &[f64], betas: &[f64]) -> Vec<Vec<f64>> {
        self.thread_pool
            .install(|| betabinom(&self.bs, &self.ns, alphas, betas))
    }

    pub fn betabinom_reduce(
        &self,
        alphas: &[f64],
        betas: &[f64],
    ) -> f64 {
        self.thread_pool
            .install(|| betabinom_reduce(&self.bs, &self.ns, alphas, betas, self.bb_weights.view()))
    }
}

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
        ws: PyReadonlyArray2<'_, f64>,
        compress: bool,
    ) -> PyResult<Self> {
        let ks = ks.as_slice()?.to_vec();
        let xs = xs.as_slice()?.to_vec();
        let bs = bs.as_slice()?.to_vec();
        let ns = ns.as_slice()?.to_vec();

        let ws = ws.as_array().to_owned();

        let inner = CnaEmission::new(ks, xs, bs, ns, ws, compress);

        Ok(CnaEmissionRs { inner })
    }

    fn nbinom(
        &self,
        py: Python,
        means: PyReadonlyArray1<'_, f64>,
        overdisp: f64,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let means = means.as_slice()?;
        let result = self.inner.nbinom(means, overdisp);
        let array = PyArray2::from_vec2(py, &result)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to create NumPy array"))?;

        Ok(array.to_owned())
    }

    fn nbinom_reduce(&self, means: PyReadonlyArray1<'_, f64>, overdisp: f64) -> PyResult<f64> {
        let means = means.as_slice()?;
        
        Ok(self.inner.nbinom_reduce(means, overdisp))
    }

    fn betabinom(
        &self,
        py: Python,
        alphas: PyReadonlyArray1<'_, f64>,
        betas: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let alphas = alphas.as_slice()?;
        let betas = betas.as_slice()?;

        let result = self.inner.betabinom(alphas, betas);

        let array = PyArray2::from_vec2(py, &result)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to create NumPy array"))?;

        Ok(array.to_owned())
    }

    fn betabinom_reduce(
        &self,
        alphas: PyReadonlyArray1<'_, f64>,
        betas: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        let alphas = alphas.as_slice()?;
        let betas = betas.as_slice()?;

        Ok(self.inner.betabinom_reduce(alphas, betas))
    }
}

//  NB  104.98 µs -> 70 µs (for all cores)
pub fn nbinom_reduce(
    k: &[f64],
    x: &[f64],
    means: &[f64],
    overdisp: f64,
    weights: ArrayView2<'_, f64>,
) -> f64 {
    let rr = 1.0 / overdisp;

    let result: f64 = k
        .par_iter()
        .zip(x.par_iter())
        .zip(weights.axis_iter(Axis(0)).into_par_iter())
        .map(|((k_val, &x_val), weights_row)| {
            let zero_point = -ln_gamma(1.0 + k_val);

            means
                .iter()
                .zip(weights_row.iter())
                .map(|(&mean_val, &weight)| {
                    let factor = 1.0 + overdisp * x_val * mean_val;

                    let ln_pp: f64 = -factor.ln();
                    let ln_qq: f64 = (1.0 - 1.0 / factor).ln();

                    let mut interim = zero_point;
                    interim += k_val * ln_qq + rr * ln_pp - ln_gamma(rr);
                    interim += ln_gamma(k_val + rr);

                    weight * interim
                })
                .sum::<f64>()
        })
        .sum();

    return result;
}

//  NB  264.86 µs -> 91.962 µs
pub fn nbinom(k: &[f64], x: &[f64], means: &[f64], overdisp: f64) -> Vec<Vec<f64>> {
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
/*
//  NB  300 µs -> 108.68 µs (all cores)
pub fn betabinom_reduce(k: &[f64], n: &[f64], a: &[f64], b: &[f64], weights: ArrayView2<'_, f64>,) -> f64 {
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
*/

pub fn betabinom_reduce(
    k: &[f64],
    n: &[f64],
    a: &[f64],
    b: &[f64],
    weights: ArrayView2<'_, f64>,
) -> f64 {
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
        .zip(weights.axis_iter(Axis(0)).into_par_iter())
        .map(|((&k_val, &n_val), weights_row)| {
            let zero_point =
                ln_gamma(n_val + 1.0) - ln_gamma(k_val + 1.0) - ln_gamma(n_val - k_val + 1.0);

            let sum: f64 = izip!(a, b, &ga, &gb, &gab, weights_row.iter())
                .map(|(&a_val, &b_val, &ga_val, &gb_val, &gab_val, &weight)| {
                    let mut interim = zero_point + gab_val - ga_val - gb_val;

                    interim += ln_gamma(k_val + a_val) + ln_gamma(n_val - k_val + b_val)
                        - ln_gamma(n_val + a_val + b_val);

                    weight * interim // Apply the weight to the computed value
                })
                .sum();

            sum
        })
        .sum();

    result
}

//  NB  125.29 µs
pub fn betabinom(k: &[f64], n: &[f64], a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
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
        .zip(n.par_iter())
        .map(|(&k_val, &n_val)| {
            let zero_point =
                ln_gamma(n_val + 1.0) - ln_gamma(k_val + 1.0) - ln_gamma(n_val - k_val + 1.0);

            let row: Vec<f64> = a
                .iter()
                .enumerate()
                .map(|(ss, &a_val)| {
                    let mut interim = zero_point;

                    interim += ln_gamma(k_val + a_val) + ln_gamma(n_val - k_val + b[ss])
                        - ln_gamma(n_val + a_val + b[ss]);
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
fn nbinom_rs(
    py: Python,
    k: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray1<'_, f64>,
    means: PyReadonlyArray1<'_, f64>,
    overdisp: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    // PyResult<Vec<Vec<f64>>>
    //
    //  Efficient negative binomial evaluation for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Negative_binomial_distribution
    let k = k.as_slice()?;
    let x = x.as_slice()?;

    let means = means.as_slice()?;
    let result = nbinom(&k, &x, &means, overdisp);

    let array = PyArray2::from_vec2(py, &result)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to create NumPy array"))?;

    Ok(array.to_owned())
}

#[pyfunction]
fn betabinom_rs(
    py: Python,
    k: PyReadonlyArray1<'_, f64>,
    n: PyReadonlyArray1<'_, f64>,
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    //
    //  Efficient beta binomial evaluation for many samples x many states.
    //
    //  see: https://en.wikipedia.org/wiki/Beta-binomial_distribution
    let k = k.to_vec()?;
    let n = n.to_vec()?;
    let a = a.to_vec()?;
    let b = b.to_vec()?;

    let result = betabinom(&k, &n, &a, &b);

    let array = PyArray2::from_vec2(py, &result)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to create NumPy array"))?;

    Ok(array.to_owned())
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
    m.add_class::<CnaEmissionRs>()?;
    m.add_function(wrap_pyfunction!(nbinom_rs, m)?)?;
    m.add_function(wrap_pyfunction!(betabinom_rs, m)?)?;
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

    #[test]
    fn test_nbinom_reduce() {
        let k = vec![1.0, 2.0, 3.0];
        let x = vec![0.5, 1.5, 2.5];

        let means = vec![1.0, 2.0, 3.0];
        let overdisp = 0.1;

        let weights = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.9, 0.7, 0.5],
            vec![0.8, 0.6, 0.4],
        ];

        let weights: Vec<f64> = weights.into_iter().flatten().collect();
        let weights = Array2::from_shape_vec((3, 3), weights).unwrap();

        let result = nbinom_reduce(&k, &x, &means, overdisp, weights.view());

        let interim = nbinom(&k, &x, &means, overdisp);
        let interim =
            Array2::from_shape_vec((3, 3), interim.into_iter().flatten().collect()).unwrap();

        let exp = (interim * &weights).sum();

        //  println!("{}  {}", result, exp);

        assert!(
            (result - exp).abs() < 1e-6,
            "result: {}, expected: {}",
            result,
            exp
        );
    }

    #[test]
    fn test_betabinom_reduce() {
        let k = vec![1.0, 2.0, 3.0];
        let n = vec![5.0, 6.0, 7.0];
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let weights = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.9, 0.7, 0.5],
            vec![0.8, 0.6, 0.4],
        ];

        let weights: Vec<f64> = weights.into_iter().flatten().collect();
        let weights = Array2::from_shape_vec((3, 3), weights).unwrap();

        let result = betabinom_reduce(&k, &n, &a, &b, weights.view());

        let interim = betabinom(&k, &n, &a, &b);
        let interim =
            Array2::from_shape_vec((3, 3), interim.into_iter().flatten().collect()).unwrap();

        let exp = (interim * &weights).sum();

        //  println!("{}  {}", result, exp);

        assert!(
            (result - exp).abs() < 1e-6,
            "result: {}, expected: {}",
            result,
            exp
        );
    }
}
