use std::{
    f64::consts::{FRAC_PI_2, PI},
    thread,
    time::Duration,
};

use libjaka::JakaMini2;
use nalgebra as na;
use robot_behavior::{Arm, MotionType, Pose, behavior::*};
use roplat_rerun::RerunHost;
use rsbullet::RsBullet; // 引入 nalgebra 处理数学计算

fn main() -> anyhow::Result<()> {
    let mut renderer = RerunHost::new("jaka_calibration")?;
    let mut physics_engine = RsBullet::new(rsbullet::Mode::Gui)?;

    //TODO:
    let translation = na::Translation3::new(0.42, 0.0, 0.0);
    let rotation = na::UnitQuaternion::from_quaternion(na::Quaternion::new(0.0, 0.0, 1.0, 0.0));
    let target_pose = na::Isometry3::from_parts(translation, rotation);

    physics_engine
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -9.81])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;
    renderer.add_search_path("./asserts")?;

    let mut robot = physics_engine
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

    for _ in 0..100 {
        physics_engine.step()?;
    }

    robot.move_joint(&[0.0, -FRAC_PI_2, 0.0, 0.0, -FRAC_PI_2, 0.0])?;

    for _ in 0..200 {
        physics_engine.step()?;
    }

    // for _ in 0..10 {
    //     physics_engine.step()?;
    // }
    // let _ = robot.state()?;

    robot.move_cartesian(&Pose::Quat(target_pose))?;
    //robot.move_joint(&[FRAC_PI_2; 6])?;

    for _ in 0..1000 {
        physics_engine.step()?;
    }

    //  测试画一条短线
    // println!(">>> Drawing a line along +Y axis...");

    // let line_end_pose = na::Isometry3::from_parts(
    //     na::Translation3::new(0.4, 0.1, 0.2),
    //     rotation,
    // );
    // robot.move_cartesian(&Pose::Quat(line_end_pose))?;

    loop {
        physics_engine.step()?;
        // 1. 获取机器人状态
        let current_state = robot.state()?;

        // 2. 提取末端位姿 (Pose Origin to End-Effector)
        if let Some(pose) = current_state.pose_o_to_ee {
            match pose {
                // 根据源码，JakaRobot 的实现通常返回 Pose::Euler
                Pose::Euler(trans, rot) => {
                    let x = trans[0];
                    let y = trans[1];
                    let z = trans[2];

                    println!("📍 末端坐标: X={:.4}, Y={:.4}, Z={:.4}", x, y, z);

                    // 注意：如果在 rsbullet 仿真中，单位通常是 米 (m)
                    // 如果是真机连接，JAKA 原始数据通常是 毫米 (mm)，但要注意库是否做了转换
                }
                // 如果返回的是四元数格式 (Isometry3)
                Pose::Quat(iso) => {
                    let x = iso.translation.vector.x;
                    let y = iso.translation.vector.y;
                    let z = iso.translation.vector.z;
                    println!("📍 末端坐标(Quat): X={:.4}, Y={:.4}, Z={:.4}", x, y, z);
                }
                _ => println!("其他位姿格式: {:?}", pose),
            }
        }
    }
}
