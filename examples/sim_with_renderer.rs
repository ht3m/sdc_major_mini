use std::{f64::consts::FRAC_PI_2, time::Duration};

use libjaka::JakaMini2;
use robot_behavior::behavior::*;
use roplat_rerun::RerunHost;
use rsbullet::{Mode, RsBullet};

fn main() -> anyhow::Result<()> {
    let mut physics = RsBullet::new(Mode::Gui)?;
    let mut renderer = RerunHost::new("mini_exam")?;

    physics
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -10.])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;
    renderer.add_search_path("./asserts")?;

    let mut robot = physics
        .robot_builder::<JakaMini2>("exam_robot")
        .base([0., 0., 0.])
        .load()?;

    let robot_render = renderer
        .robot_builder("exam_robot")
        .base([0., 0., 0.])
        .load()?;

    robot_render.attach_from(&mut robot)?;

    robot.move_joint(&[FRAC_PI_2; 6])?;

    loop {
        physics.step()?;
    }
}
