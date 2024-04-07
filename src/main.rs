use rayon::prelude::*;
use peroxide::fuga::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Data generation - Gaussian mixtures
    let n_1 = Normal(1.0, 0.2);
    let n_2 = Normal(3.0, 0.5);
    let true_pdf = |x: f64| n_1.pdf(x) + n_2.pdf(x);
    let data = concat(&n_1.sample(800), &n_2.sample(700));

    // Kernel Density Estimation
    let x = linspace(0, 5, 1000);
    let y = x.fmap(true_pdf);
    let bandwidth = 0.35;
    let kde = kernel_density_estimation(&data, bandwidth, Kernel::Quartic, &x);
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
    let samples = prs(f, 10000, (0.0, 5.0), 100, 1e-6);

    // Histogram from samples
    let x_bin = linspace(0, 5, 100);
    let mut n_bin = vec![0f64; x_bin.len()];
    samples.iter()
        .for_each(|x| {
            let bin_idx = (x - 0.0) / (5.0 - 0.0) * (x_bin.len() as f64);
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

    Ok(())
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

