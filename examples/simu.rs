use anyhow::Result;
use libjaka::JakaMini2;
use robot_behavior::behavior::*; // 引入 Trait
use roplat_rerun::RerunHost;
use rsbullet::RsBullet;
use std::time::Duration;

fn main() -> Result<()> {
    // 1. 初始化仿真
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

    // =============================================================
    // 2. 使用 move_traj_from_file
    // =============================================================
    println!("📂 读取并执行轨迹...");

    // 🔥 注意：这里要填那个新生成的 rust_enum.json 路径
    let file_path = "./robot_draw/img/step08_optimized_trajectory_rust.json";

    // 如果想从起点开始，最好先读取第一帧瞬移过去，
    // 但 move_traj_from_file 内部直接拿走了所有权，所以仿真里我们简单处理：
    // 让它直接开始跑，第一帧可能会有个瞬移跳跃。

    // 这一步会读取文件，解析成 Vec<MotionType>，并注册进任务队列
    robot.move_traj_from_file(file_path)?;

    println!("🚀 开始仿真循环...");

    // =============================================================
    // 3. 驱动时间 (必不可少)
    // =============================================================
    // move_traj_from_file 只是把任务放进了队列，它自己不包含循环。
    // 我们必须在这里推着物理引擎走。

    println!("✨ 轨迹执行完毕");

    loop {}
}
