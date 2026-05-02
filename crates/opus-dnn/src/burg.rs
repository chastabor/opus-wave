/// Burg analysis: compute LP prediction coefficients from input signal.
/// Matches C `silk_burg_analysis` from dnn/burg.c.
///
/// Returns residual energy. `a` receives the prediction coefficients.
pub fn silk_burg_analysis(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    order: usize,
) -> f32 {
    const MAX_ORDER: usize = 16;
    const FIND_LPC_COND_FAC: f64 = 1e-5;

    debug_assert!(order <= MAX_ORDER);

    let mut c_first_row = [0.0f64; MAX_ORDER];
    let mut c_last_row = [0.0f64; MAX_ORDER];
    let mut caf = [0.0f64; MAX_ORDER + 1];
    let mut cab = [0.0f64; MAX_ORDER + 1];
    let mut af = [0.0f64; MAX_ORDER];

    let c0 = energy_f64(x, nb_subfr * subfr_length);

    for s in 0..nb_subfr {
        let x_ptr = &x[s * subfr_length..];
        for n in 1..order + 1 {
            c_first_row[n - 1] += inner_product_f64(x_ptr, &x_ptr[n..], subfr_length - n);
        }
    }
    c_last_row[..MAX_ORDER].copy_from_slice(&c_first_row[..MAX_ORDER]);

    caf[0] = c0 + FIND_LPC_COND_FAC * c0 + 1e-9;
    cab[0] = caf[0];
    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;

    for n in 0..order {
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            let mut tmp1 = x_ptr[n] as f64;
            let mut tmp2 = x_ptr[subfr_length - n - 1] as f64;
            for k in 0..n {
                c_first_row[k] -= (x_ptr[n] as f64) * (x_ptr[n - k - 1] as f64);
                c_last_row[k] -=
                    (x_ptr[subfr_length - n - 1] as f64) * (x_ptr[subfr_length - n + k] as f64);
                let atmp = af[k];
                tmp1 += (x_ptr[n - k - 1] as f64) * atmp;
                tmp2 += (x_ptr[subfr_length - n + k] as f64) * atmp;
            }
            for k in 0..=n {
                caf[k] -= tmp1 * (x_ptr[n - k] as f64);
                cab[k] -= tmp2 * (x_ptr[subfr_length - n + k - 1] as f64);
            }
        }

        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }

        let mut rc = -2.0 * num / (nrg_f + nrg_b);

        let tmp1_gain = inv_gain * (1.0 - rc * rc);
        if tmp1_gain <= min_inv_gain as f64 {
            rc = (1.0 - (min_inv_gain as f64) / inv_gain).sqrt();
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain as f64;
            reached_max_gain = true;
        } else {
            inv_gain = tmp1_gain;
        }

        for k in 0..(n + 1) >> 1 {
            let t1 = af[k];
            let t2 = af[n - k - 1];
            af[k] = t1 + rc * t2;
            af[n - k - 1] = t2 + rc * t1;
        }
        af[n] = rc;

        if reached_max_gain {
            for v in af.iter_mut().take(order).skip(n + 1) {
                *v = 0.0;
            }
            break;
        }

        for k in 0..=n + 1 {
            let t1 = caf[k];
            caf[k] += rc * cab[n + 1 - k];
            cab[n + 1 - k] += rc * t1;
        }
    }

    let nrg_f;
    if reached_max_gain {
        for k in 0..order {
            a[k] = -af[k] as f32;
        }
        let mut c0_adj = c0;
        for s in 0..nb_subfr {
            c0_adj -= energy_f64(&x[s * subfr_length..], order);
        }
        nrg_f = c0_adj * inv_gain;
    } else {
        let mut nrg = caf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..order {
            let atmp = af[k];
            nrg += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = -atmp as f32;
        }
        nrg_f = nrg - FIND_LPC_COND_FAC * c0 * tmp1;
    }

    (nrg_f as f32).max(0.0)
}

fn energy_f64(data: &[f32], len: usize) -> f64 {
    let mut result = 0.0f64;
    let mut i = 0;
    while i < len.saturating_sub(3) {
        result += data[i] as f64 * data[i] as f64
            + data[i + 1] as f64 * data[i + 1] as f64
            + data[i + 2] as f64 * data[i + 2] as f64
            + data[i + 3] as f64 * data[i + 3] as f64;
        i += 4;
    }
    while i < len {
        result += data[i] as f64 * data[i] as f64;
        i += 1;
    }
    result
}

fn inner_product_f64(data1: &[f32], data2: &[f32], len: usize) -> f64 {
    let mut result = 0.0f64;
    let mut i = 0;
    while i < len.saturating_sub(3) {
        result += data1[i] as f64 * data2[i] as f64
            + data1[i + 1] as f64 * data2[i + 1] as f64
            + data1[i + 2] as f64 * data2[i + 2] as f64
            + data1[i + 3] as f64 * data2[i + 3] as f64;
        i += 4;
    }
    while i < len {
        result += data1[i] as f64 * data2[i] as f64;
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_burg_dc_signal() {
        // DC signal should have near-zero prediction coefficients beyond a[0].
        let signal = vec![1.0f32; 80];
        let mut a = [0.0f32; 4];
        let energy = silk_burg_analysis(&mut a, &signal, 1e-3, 80, 1, 4);
        // First coefficient should be close to 1 (predict x[n] from x[n-1])
        assert!(a[0].abs() > 0.9, "a[0]={}", a[0]);
        assert!(energy >= 0.0);
    }

    #[test]
    fn test_burg_zero_signal() {
        let signal = vec![0.0f32; 80];
        let mut a = [0.0f32; 4];
        let energy = silk_burg_analysis(&mut a, &signal, 1e-3, 80, 1, 4);
        // Energy is near-zero (only the conditioning factor 1e-9 remains)
        assert!(energy < 1e-6, "energy = {energy}");
    }

    #[test]
    fn test_burg_returns_positive_energy() {
        let signal: Vec<f32> = (0..160).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut a = [0.0f32; 8];
        let energy = silk_burg_analysis(&mut a, &signal, 1e-3, 80, 2, 8);
        assert!(energy >= 0.0);
    }
}
