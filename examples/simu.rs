use anyhow::{Context, Result};
use libjaka::JakaMini2;
use robot_behavior::{MotionType, Robot, behavior::*};
use roplat_rerun::RerunHost;
use rsbullet::RsBullet;
use std::fs::File;
use std::io::BufReader;
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

    let file_path = "./robot_draw/img/step09_stable_traj_rust.json";

    let file = File::open(file_path).context("无法打开轨迹文件")?;
    let reader = BufReader::new(file);
    let trajectory: Vec<MotionType<6>> = serde_json::from_reader(reader)?;
    if trajectory.is_empty() {
        println!("⚠️ 轨迹为空，程序退出");
        return Ok(());
    }

    if let Some(MotionType::Joint(start_pose)) = trajectory.first() {
        println!("📍 瞬移到起始姿态...");
        robot.move_joint(start_pose)?;
    }

    for _ in 0..125 {
        physics.step()?;
    }

    robot.move_traj_from_file(file_path)?;

    println!("🚀 开始仿真循环...");

    loop {
        physics.step()?;
    }
}
