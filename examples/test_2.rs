use std::{
    f64::consts::{FRAC_PI_2, FRAC_PI_4},
    time::Duration,
};

use libjaka::JakaMini2;
use nalgebra as na;
use robot_behavior::{Pose, behavior::*};
use roplat_rerun::RerunHost;
use rsbullet::RsBullet;

fn main() -> anyhow::Result<()> {
    let mut renderer = RerunHost::new("jaka_dual")?;
    let mut physics_engine = RsBullet::new(rsbullet::Mode::Gui)?;

    physics_engine
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -10.])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;
    renderer.add_search_path("./asserts")?;

    let mut robot_1 = physics_engine
        .robot_builder::<JakaMini2>("robot_1")
        .base([0.0, 0.0, 0.0])
        .base_fixed(true)
        .load()?;

    let robot_1_renderer = renderer
        .robot_builder::<JakaMini2>("robot_1")
        .base([0.0, 0.0, 0.0])
        .base_fixed(true)
        .load()?;
    robot_1_renderer.attach_from(&mut robot_1)?;

    for _ in 0..100 {
        physics_engine.step()?;
    }
    robot_1.move_joint(&[ 0.0200, -0.7216, -1.5476, -0.0000, -0.8724, 0.0200])?;
    // robot_1.move_joint(&[0.0; 6])?;
    for _ in 0..500 {
        physics_engine.step()?;
    }
    loop {
        physics_engine.step()?;
    }
}
