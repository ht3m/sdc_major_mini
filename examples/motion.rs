use libjaka::JakaMini2;
use nalgebra as na;
use robot_behavior::{MotionType, Pose, behavior::*};
use rsbullet::RsBulletRobot;

fn main() -> anyhow::Result<()> {
    let mut robot: RsBulletRobot<JakaRobot<JakaMini2, 6>> = JakaMini2::new("10.5.5.100");

    robot.move_joint(&[0.; 6])?;
    robot.move_to(MotionType::Joint([0.; 6]))?;

    robot.move_cartesian(&Pose::Quat(na::Isometry3::identity()))?;
    robot.move_to(MotionType::Cartesian(Pose::Quat(na::Isometry3::identity())))?;

    // 使用 `xx.json` 文件中的轨迹进行运动
    // 文件内要求轨迹形式为：
    // [
    //   { "Cartesian": {"Euler": [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]] }},
    // ]
    robot.move_traj_from_file("xx.json")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use libjaka::JakaMini2;
    use robot_behavior::behavior::*;

    #[test]
    fn move_to_default() -> RobotResult<()> {
        let mut robot = JakaMini2::new("10.5.5.100");
        robot.move_joint(&JakaMini2::JOINT_DEFAULT)?;
        Ok(())
    }

    #[test]
    fn move_to_packed() -> RobotResult<()> {
        let mut robot = JakaMini2::new("10.5.5.100");
        robot.move_joint(&JakaMini2::JOINT_PACKED)?;
        Ok(())
    }
}
