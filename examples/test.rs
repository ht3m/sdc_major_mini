use std::{f64::consts::FRAC_PI_2, f64::consts::FRAC_PI_4, time::Duration};

use libjaka::JakaMini2;
use robot_behavior::behavior::*;
use roplat_rerun::RerunHost;
use rsbullet::RsBullet;

fn main() -> anyhow::Result<()> {
    let mut renderer = RerunHost::new("jaka_dual")?;
    let mut physics_engine = RsBullet::new(rsbullet::Mode::Gui)?;

    physics_engine
        .add_search_path("F:\\2025 Autumn\\sdc_major_workspace\\sdc_major\\asserts")?
        .set_gravity([0., 0., -10.])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;
    renderer.add_search_path("F:\\2025 Autumn\\sdc_major_workspace\\sdc_major\\asserts")?;

    let mut robot_1 = physics_engine
        .robot_builder::<JakaMini2>("robot_1")
        .base([0.0, 0.2, 0.0])
        .base_fixed(true)
        .load()?;
    let mut robot_2 = physics_engine
        .robot_builder::<JakaMini2>("robot_2")
        .base([0.0, -0.2, 0.0])
        .base_fixed(true)
        .load()?;

    let robot_1_renderer = renderer
        .robot_builder::<JakaMini2>("robot_1")
        .base([0.0, 0.2, 0.0])
        .base_fixed(true)
        .load()?;
    let robot_2_renderer = renderer
        .robot_builder::<JakaMini2>("robot_2")
        .base([0.0, -0.2, 0.0])
        .base_fixed(true)
        .load()?;
    robot_1_renderer.attach_from(&mut robot_1)?;
    robot_2_renderer.attach_from(&mut robot_2)?;

    for _ in 0..100 {
        physics_engine.step()?;
    }
    robot_1.move_joint(&[FRAC_PI_2; 6])?;
    robot_2.move_joint(&[FRAC_PI_4; 6])?;
    loop {
        physics_engine.step()?;
    }
}
