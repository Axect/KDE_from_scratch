use rayon::prelude::*;
use peroxide::fuga::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let left_peak = 1.0;
    let right_peak = 3.0;
    let left_most = 0.0;
    let right_most = 5.0;

    // Data generation - Gaussian mixtures
    let n_1 = Normal(left_peak, 0.2);
    let n_2 = Normal(right_peak, 0.5);
    let true_pdf = |x: f64| n_1.pdf(x) + n_2.pdf(x);
    let data = concat(&n_1.sample(800), &n_2.sample(700));

    // Kernel Density Estimation
    let x = linspace(left_most, right_most, 1000);
    let y = x.fmap(true_pdf);
    let kernel = Kernel::Epanechnikov;
    let bandwidth = silverman_bandwidth(&data);
    bandwidth.print();
    let kde = kernel_density_estimation(&data, bandwidth, kernel, &x);
    let scale_factor = y.max() / kde.max();
    let kde = kde.fmap(|x| x * scale_factor);

    // Cubic Hermite Spline
    let cs = cubic_hermite_spline(&x, &kde, Quadratic);
    let f = |x: f64| {
        let y = cs.eval(x);
        if y < 0.0 {
            0.0
        } else {
            y
        }
    };

    // Piecewise Rejection Sampling
    let samples = prs(f, 10000, (left_most, right_most), 100, 1e-6);

    // Histogram from samples
    let x_bin = linspace(left_most, right_most, 100);
    let mut n_bin = vec![0f64; x_bin.len()];
    samples.iter()
        .for_each(|x| {
            let bin_idx = (x - left_most) / (right_most - left_most) * (x_bin.len() as f64);
            n_bin[bin_idx as usize] += 1.0;
        });
    let y_bin = x_bin.fmap(true_pdf);
    let scale_factor = y_bin.max() / n_bin.max();
    let n_bin = n_bin.fmap(|x| x * scale_factor);

    // Plot
    let mut plt = Plot2D::new();
    plt
        .set_domain(x)
        .insert_image(y)
        .insert_image(kde)
        .set_legend(vec!["True", "KDE"])
        .set_line_style(vec![(0, LineStyle::Solid), (1, LineStyle::Dashed)])
        .set_color(vec![(0, "black"), (1, "red")])
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("kde.png")
        .savefig()?;

    let mut plt = Plot2D::new();
    plt
        .set_domain(x_bin)
        .insert_image(y_bin)
        .insert_image(n_bin)
        .set_legend(vec!["True", "Histogram"])
        .set_plot_type(vec![(0, PlotType::Line), (1, PlotType::Scatter)])
        .set_marker(vec![(1, Markers::Point)])
        .set_color(vec![(0, "black"), (1, "red")])
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("histogram.png")
        .savefig()?;

    //// TPE
    //let mut tpe = TPEFloat::new((0f64, 6f64), 0.2, 100, 10, Kernel::Epanechnikov);
    //for _ in 0 .. 100 {
    //    let param_candidates = tpe.ask();
    //    param_candidates.print();
    //    let param_max = param_candidates.max();
    //    let param_min = param_candidates.min();
    //    if param_max - param_min < 1e-3 * 10f64 {
    //        break;
    //    }
    //    param_candidates.into_iter()
    //        .for_each(
    //            |param| {
    //                let object = objective(param);
    //                tpe.report(param, object);
    //            }
    //        );
    //}
    //let best_params = tpe.ask();
    //best_params.print();

    Ok(())
}

fn objective(x: f64) -> f64 {
    x.sin().abs()
}

/// Gaussian kernel function
fn gaussian_kernel(x: f64, bandwidth: f64) -> f64 {
    (-x.powi(2) / (2.0 * bandwidth.powi(2))).exp() / (bandwidth * (2.0 * std::f64::consts::PI).sqrt())
}

/// Epanechnikov kernel function
fn epanechnikov_kernel(x: f64, bandwidth: f64) -> f64 {
    if x.abs() <= bandwidth {
        0.75 * (1.0 - x.powi(2) / bandwidth.powi(2))
    } else {
        0.0
    }
}

/// Tri-Cube kernel function
fn tricube_kernel(x: f64, bandwidth: f64) -> f64 {
    if x.abs() <= bandwidth {
        70f64/81f64 * (1.0 - x.abs().powi(3) / bandwidth.powi(3)).powi(3)
    } else {
        0.0
    }
}

/// Quartic kernel function
fn quartic_kernel(x: f64, bandwidth: f64) -> f64 {
    if x.abs() <= bandwidth {
        15f64/16f64 * (1.0 - x.powi(2) / bandwidth.powi(2)).powi(2)
    } else {
        0.0
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Kernel {
    Epanechnikov,
    Gaussian,
    TriCube,
    Quartic,
}

impl Kernel {
    fn pdf(self, x: f64, bandwidth: f64) -> f64 {
        match self {
            Kernel::Epanechnikov => epanechnikov_kernel(x, bandwidth),
            Kernel::Gaussian => gaussian_kernel(x, bandwidth),
            Kernel::TriCube => tricube_kernel(x, bandwidth),
            Kernel::Quartic => quartic_kernel(x, bandwidth),
        }
    }
}

/// 1D Kernel Density Estimation
fn kernel_density_estimation(data: &[f64], bandwidth: f64, kernel: Kernel, domain: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;

    domain
        .to_vec()
        .fmap(|x| {
            let kde = data
                .par_iter()
                .fold(|| 0.0, |kde, &x_i| kde + kernel.pdf(x - x_i, bandwidth))
                .sum::<f64>();
            kde / n
        })
}

/// Scott's rule for estimating bandwidth
#[allow(dead_code)]
fn scott_rule(data: &[f64]) -> f64 {
    let data = data.to_vec();
    let n = data.len() as f64;
    let std = data.sd();
    1.06 * std * n.powf(-0.2)
}

/// Silverman's rule of thumb for Gaussian kernel
#[allow(dead_code)]
fn silverman_bandwidth(data: &[f64]) -> f64 {
    let data = data.to_vec();
    let n = data.len() as f64;
    let sigma = data.sd();
    (4.0 * sigma.powi(5) / (3.0 * n)).powf(1.0 / 5.0)
}

/// Optuna's bandwidth
#[allow(dead_code)]
fn optuna_bandwidth(data: &[f64], left: f64, right: f64) -> f64 {
    (right - left) / 5f64 * (data.len() as f64).powf(-0.2)
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TPEFloat {
    hyperparameter_range: (f64, f64),
    alpha: f64,
    n_init_samples: usize,
    n_out_samples: usize,
    kernel: Kernel,
    evaluations: Vec<(f64, f64)>,
}

impl TPEFloat {
    fn new(hyperparameter_range: (f64, f64), alpha: f64, n_init_samples: usize, n_out_samples: usize, kernel: Kernel) -> Self {
        TPEFloat {
            hyperparameter_range,
            alpha,
            n_init_samples,
            n_out_samples,
            kernel,
            evaluations: Vec::new(),
        }
    }

    fn get_range(&self) -> (f64, f64) {
        self.hyperparameter_range
    }

    fn report(&mut self, hyperparameter: f64, score: f64) {
        self.evaluations.push((hyperparameter, score));
    }

    fn ask(&mut self) -> Vec<f64> {
        // TPE 최적화 로직 구현
        // 1. 초기 하이퍼파라미터 값 랜덤 샘플링
        // 2. 반복:
        //    - self.evaluations를 성능에 따라 good, bad 그룹으로 분할
        //    - good, bad 각각에 대해 KDE로 분포 추정
        //    - l(x) / g(x)를 최대화하는 하이퍼파라미터 값 선택
        //    - 선택된 값을 반환하여 사용자가 평가 후 report로 결과 전달
        // 3. 최적 하이퍼파라미터 값 반환
        if self.evaluations.len() < self.n_init_samples {
            let u = Uniform(self.hyperparameter_range.0, self.hyperparameter_range.1);
            u.sample(self.n_out_samples)
        } else {
            let (left, right) = self.get_range();
            let mut eval = self.evaluations.clone();
            eval.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let good_len = (eval.len() as f64 * self.alpha) as usize;

            let (good, _): (Vec<f64>, Vec<f64>) = eval[..good_len].iter().cloned().unzip();
            let (bad, _): (Vec<f64>, Vec<f64>) = eval[good_len..].iter().cloned().unzip();

            let bandwidth_good = silverman_bandwidth(&good);
            let bandwidth_bad = silverman_bandwidth(&bad);

            let domain = linspace(left, right, 100);

            let kde_good = kernel_density_estimation(&good, bandwidth_good, self.kernel, &domain);
            let kde_bad = kernel_density_estimation(&bad, bandwidth_bad, self.kernel, &domain);

            let ei = zip_with(|l, g| {
                let g = if g <= 1e-6 { 1e-6 } else { g };
                let l = if l <= 0f64 { 0f64 } else { l };
                l / g
            }, &kde_good, &kde_bad);

            let cs_ei = cubic_hermite_spline(&domain, &ei, Quadratic);

            let f = |x: f64| { 
                let y = cs_ei.eval(x);
                if y < 0.0 {
                    0.0
                } else {
                    y
                }
            };

            prs(f, self.n_out_samples, (left, right), 10, 1e-6)
        }
    }
}
