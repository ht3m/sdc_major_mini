use std::{fs::File, io::BufReader, path::Path, time::Duration};

use anyhow::Context;
use libjaka::JakaMini2;
use robot_behavior::{
    ArmPreplannedMotion, // <--- 包含 move_joint_async
    Robot,               // <--- 包含 is_moving
    behavior::*,         // 基础行为
};
use roplat_rerun::RerunHost;
use rsbullet::RsBullet;
use serde::Deserialize;
use std::convert::TryInto;

// 定义数据结构 (对应 step08_optimized_trajectory.json)
#[derive(Deserialize, Debug)]
struct TrajectoryData {
    #[allow(dead_code)]
    meta: serde::de::IgnoredAny,
    joints: Vec<Vec<f64>>,
}

fn main() -> anyhow::Result<()> {
    // =============================================================
    // 1. 初始化仿真环境
    // =============================================================
    let mut renderer = RerunHost::new("jaka_dual")?;
    let mut physics = RsBullet::new(rsbullet::Mode::Gui)?;

    physics
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -9.8])?
        // 🔥 关键：设置物理步长为 125Hz (8ms)
        .set_step_time(Duration::from_secs_f64(1. / 125.))?;
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

    // =============================================================
    // 2. 加载轨迹文件
    // =============================================================
    let json_path = Path::new("./robot_draw/img/step08_optimized_trajectory.json");
    println!("📂 加载轨迹: {:?}", json_path);

    let file = File::open(json_path).context("找不到 JSON 文件")?;
    let reader = BufReader::new(file);
    let trajectory: TrajectoryData = serde_json::from_reader(reader)?;

    let points = trajectory.joints;
    if points.is_empty() {
        println!("⚠️ 警告: 轨迹数据为空");
        return Ok(());
    }
    println!("✅ 轨迹加载完成，共 {} 帧 (每帧 8ms)", points.len());

    // =============================================================
    // 3. 移动到起点 (初始化)
    // =============================================================
    let start_vec = &points[0];
    let start_arr: [f64; 6] = start_vec.as_slice().try_into().unwrap();

    println!("🚀 移动到起始姿态...");
    robot.move_joint_async(&start_arr)?;

    // 预热 1 秒 (125 * 8ms)
    for _ in 0..125 {
        physics.step()?;
    }

    // =============================================================
    // 4. 手动执行流式轨迹 (Manual Streaming)
    // =============================================================
    println!("🎨 开始绘制 ({} 帧)...", points.len());

    // 我们不使用 move_traj，而是手动循环
    // 这样可以确保每一帧数据对应物理引擎的一次 step
    for (i, point_vec) in points.iter().enumerate() {
        // 1. 数据转换 Vec -> Array
        let target: [f64; 6] = point_vec.as_slice().try_into().expect("关节数据错误");

        // 2. 更新目标点 (PTP Setpoint)
        // 在仿真中，move_joint_async 只是告诉物理引擎：“下一个目标是这里”
        // 只要我们调用的频率够快（每次 step 调一次），它就是流式控制
        robot.move_joint_async(&target)?;

        // 3. 推进物理时间 (8ms)
        physics.step()?;
    }

    println!("✨ 绘制完成！");

    // =============================================================
    // 5. 保持窗口常驻
    // =============================================================
    loop {
        physics.step()?;
    }
}
