use std::{fs::File, io::BufReader, path::Path, time::Duration};

use anyhow::Context;
use libjaka::JakaMini2;
use nalgebra as na;
use robot_behavior::{Pose, behavior::*};
use roplat_rerun::RerunHost;
use rsbullet::RsBullet;
use serde::Deserialize;
use std::convert::TryInto;

// 定义与 step07_joint_trajectory.json 匹配的数据结构
#[derive(Deserialize, Debug)]
struct TrajectoryMeta {
    source: String,
    unit: String,
    count: usize,
}

#[derive(Deserialize, Debug)]
struct TrajectoryData {
    #[allow(dead_code)] // 仿真中可能用不到 meta，忽略未使用的警告
    meta: TrajectoryMeta,
    joints: Vec<Vec<f64>>,
}

fn main() -> anyhow::Result<()> {
    // =============================================================
    // 1. 初始化仿真环境 (Initialize)
    // =============================================================
    let mut renderer = RerunHost::new("jaka_dual")?;
    let mut physics_engine = RsBullet::new(rsbullet::Mode::Gui)?;

    physics_engine
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -9.8])?
        // 物理步长设为 240Hz，这是 PyBullet 的标准频率
        .set_step_time(Duration::from_secs_f64(1. / 125.))?;
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

    // =============================================================
    // 2. 读取 JSON 轨迹文件
    // =============================================================
    // 请确保路径正确，这里指向 step07 生成的文件
    let json_path = Path::new("./robot_draw/img/step07_more_optimized_trajectory.json");
    println!("正在加载轨迹文件: {:?}", json_path);

    let file = File::open(json_path).context("无法打开 JSON 文件，请检查路径")?;
    let reader = BufReader::new(file);
    let trajectory: TrajectoryData = serde_json::from_reader(reader)?;

    let points = trajectory.joints;
    if points.is_empty() {
        println!("警告: 轨迹数据为空");
        return Ok(());
    }
    println!("轨迹加载成功，共 {} 个点", points.len());

    // =============================================================
    // 3. 移动到起点并等待 (Move to Start & Wait)
    // =============================================================
    let start_joint = &points[0];

    // 【修改点 1】：将 Vec 转换为 [f64; 6]
    // try_into() 尝试转换，如果长度不对会报错
    let start_array: [f64; 6] = start_joint
        .as_slice()
        .try_into()
        .context("起始点关节数据长度必须为6")?;

    println!("移动到起始姿态...");
    // 传入数组的引用
    robot_1.move_joint(&start_array)?;

    for _ in 0..480 {
        physics_engine.step()?;
    }
    println!("起始姿态已稳定，开始执行轨迹...");

    // =============================================================
    // 4. 执行整个轨迹 (Run Trajectory)
    // =============================================================
    for (i, joint_target) in points.iter().enumerate().skip(1) {
        // 【修改点 2】：同样进行类型转换
        // 这里使用 unwrap() 或者 expect()，因为我们在 step07 保证了数据生成是对的
        // 如果这里报错，说明 JSON 数据坏了
        let target_array: [f64; 6] = joint_target
            .as_slice()
            .try_into()
            .expect("轨迹点关节数据长度错误");

        // 传入数组的引用
        robot_1.move_joint(&target_array)?;

        physics_engine.step()?;

        if i % 100 == 0 {
            // println!("执行进度: {}/{}", i, points.len());
        }
    }

    println!("轨迹执行完毕！保持仿真窗口开启。");

    // =============================================================
    // 5. 保持窗口常驻
    // =============================================================
    loop {
        physics_engine.step()?;
    }
}
