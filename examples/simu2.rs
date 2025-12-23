use anyhow::{Context, Result};
use serde::Deserialize;
use std::convert::TryInto;
use std::{
    fs::File,
    io::BufReader,
    path::Path,
    thread,
    time::{Duration, Instant},
};

// 假设这是你的真机驱动库
// use libjaka::JakaMini2;

// 为了代码能跑，我这里模拟一个 JakaMini2 的接口结构
// 你实际使用时，请替换回 use libjaka::JakaMini2;
struct JakaMini2;
impl JakaMini2 {
    pub fn new(_ip: &str) -> Result<Self> {
        Ok(Self)
    }

    // 基础 PTP 运动 (移动到起点用)
    pub fn move_joint(&self, _joints: &[f64; 6]) -> Result<()> {
        // 实际 SDK: robot.move_joint(...)
        println!("🤖 正在移动到初始位置...");
        thread::sleep(Duration::from_secs(2)); // 模拟耗时
        Ok(())
    }

    // 开启伺服模式 (必须)
    pub fn servo_enable(&self) -> Result<()> {
        println!("⚡ 开启 Servo 模式");
        Ok(())
    }

    // 关闭伺服模式 (必须)
    pub fn servo_disable(&self) -> Result<()> {
        println!("💤 关闭 Servo 模式");
        Ok(())
    }

    // 核心：单帧伺服指令
    // 注意：这里不是 move_joint (PTP)，而是 servo_j
    pub fn servo_j(&self, _joints: &[f64; 6]) -> Result<()> {
        // 实际 SDK: robot.servo_j(...)
        Ok(())
    }
}

// 定义数据结构
#[derive(Deserialize, Debug)]
struct TrajectoryData {
    #[allow(dead_code)]
    meta: serde::de::IgnoredAny,
    joints: Vec<Vec<f64>>,
}

fn main() -> Result<()> {
    // =============================================================
    // 1. 连接真机
    // =============================================================
    // 替换为真实 IP
    let robot = JakaMini2::new("192.168.1.100")?;
    println!("✅ 机器人连接成功");

    // =============================================================
    // 2. 加载轨迹文件
    // =============================================================
    let json_path = Path::new("./robot_draw/img/step08_optimized_trajectory.json");
    let file = File::open(json_path).context("找不到 JSON 文件")?;
    let reader = BufReader::new(file);
    let trajectory: TrajectoryData = serde_json::from_reader(reader)?;

    let points = trajectory.joints;
    if points.is_empty() {
        return Ok(());
    }

    println!("📂 轨迹加载完成: {} 帧", points.len());

    // =============================================================
    // 3. 安全复位 (Move to Start)
    // =============================================================
    // 在开启伺服模式前，机器人必须实际上已经位于轨迹的第一个点
    // 否则伺服开启瞬间，机器人会因为误差过大而急停或猛冲
    let start_vec = &points[0];
    let start_arr: [f64; 6] = start_vec.as_slice().try_into().unwrap();

    println!("🚀 [PTP] 慢速移动到轨迹起点...");
    robot.move_joint(&start_arr)?;
    println!("✅ 已到达起点，准备开始轨迹流...");

    // =============================================================
    // 4. 执行 Move Traj (流式控制)
    // =============================================================
    move_traj(&robot, &points)?;

    println!("✨ 绘制完成！");
    Ok(())
}

/// 核心函数：以严格的 125Hz 发送数据
fn move_traj(robot: &JakaMini2, points: &[Vec<f64>]) -> Result<()> {
    // 1. 开启伺服模式
    robot.servo_enable()?;

    // 定义周期 8ms
    let period = Duration::from_secs_f64(1.0 / 125.0);

    // 2. 实时循环
    for (i, point_vec) in points.iter().enumerate() {
        // --- 计时开始 ---
        let start = Instant::now();

        let target: [f64; 6] = point_vec.as_slice().try_into().expect("数据异常");

        // --- 发送指令 (非阻塞) ---
        // 这里调用的必须是 servo_j，它只是把数据塞给底层控制卡，瞬间完成
        robot.servo_j(&target)?;

        // --- 智能休眠 (Smart Sleep) ---
        // 计算发送指令消耗了多少时间
        let elapsed = start.elapsed();

        if elapsed < period {
            // 如果耗时小于 8ms，就睡够剩下的时间
            thread::sleep(period - elapsed);
        } else {
            // 如果耗时超过 8ms，说明系统延迟高，不能睡了，直接发下一帧
            // 可以在这里加个计数器，如果连续超时太多帧就报错
            // eprintln!("⚠️ 警告: 第 {} 帧超时 (耗时 {:?})", i, elapsed);
        }
    }

    // 3. 结束，关闭伺服模式
    robot.servo_disable()?;
    Ok(())
}
