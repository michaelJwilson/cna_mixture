import numpy as np
from cna_mixture_rs.core import nbinom_logpmf as nbinom_logpmf_rs
from scipy.stats import nbinom

def prepare_data(size=10_000):
    k = np.arange(1, 1 + size, dtype=np.float64)
    r = np.arange(1, 1 + size, dtype=np.float64) * 0.5
    
    p = np.full(size, 0.5, dtype=np.float64)
    
    return k, r, p

# NB 108 ms, 90ms,
def nbinom_logpmf(ks, rs, ps, RUST_BACKEND=True):
    if RUST_BACKEND:
        result = nbinom_logpmf_rs(ks, rs, ps)
    else:
        result = np.zeros(shape=(len(ks), len(rs)))

        for ss, (r, p) in enumerate(zip(rs, ps)):
            for ii, k in enumerate(ks):
                result[ii, ss] = nbinom.logpmf(k, r, p)

        result = result.sum()
                
    return result

def test_nbinom_logpmf_benchmark(benchmark):
    ks, rs, ps = prepare_data()

    benchmark(nbinom_logpmf, ks, rs, ps)

if __name__ == "__main__":
    ks, rs, ps = prepare_data()
    result = nbinom_logpmf_rs(ks, rs, ps)

    print(result)
