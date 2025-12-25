use anyhow::Result;
use libjaka::JakaMini2;
use robot_behavior::behavior::*;
use roplat_rerun::RerunHost;
use rsbullet::RsBullet;
use std::time::Duration;

fn main() -> Result<()> {
    let mut renderer = RerunHost::new("jaka_sim")?;
    let mut physics = RsBullet::new(rsbullet::Mode::Gui)?;

    physics
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -9.8])?
        .set_step_time(Duration::from_secs_f64(1.0 / 125.0))?; // 8ms

    renderer.add_search_path("./asserts")?;

    let mut robot = physics
        .robot_builder::<JakaMini2>("robot_1")
        .base([0.0, 0.0, 0.0])
        .base_fixed(true)
        .load()?;

    let robot_renderer = renderer
        .robot_builder::<JakaMini2>("robot_1")
        .base([0.0, 0.0, 0.0])
        .base_fixed(true)
        .load()?;

    robot_renderer.attach_from(&mut robot)?;

    println!("📂 读取并执行轨迹...");

    let file_path = "./robot_draw/img/step08_optimized_trajectory_rust.json";

    robot.move_traj_from_file(file_path)?;

    println!("🚀 开始仿真循环...");

    loop {}
}
