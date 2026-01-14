use libjaka::JakaMini2;
use robot_behavior::behavior::*;
use std::f64::consts::FRAC_PI_6;

fn main() -> anyhow::Result<()> {
    let mut robot = JakaMini2::new("192.168.1.1");

    // 完全等同于内置函数 robot._power_on()?;
    robot.init()?;
    robot.enable()?;

    // 此时你才可以发送运动指令
    //TODO:设置运动
    robot.move_joint(&[FRAC_PI_6; 6])?;
   // robot.move_to(MotionType::Joint([0.; 6]))?;

    robot.disable()?;
    // 完全等同于内置函数 robot._power_off()?;
    robot.shutdown()?;

    Ok(())
}

/// 一些测试函数可以使用快捷按钮，这使得开发一些临时使用的功能变得更加方便。你可以将其复制到任何部分，
/// 但是只有在 #[cfg(test)] 下才会被编译。
#[cfg(test)]
mod tests {
    use robot_behavior::behavior::*;
    #[test]
    fn power_on() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("192.168.1.1");
        robot.init()?;
        Ok(())
    }

    #[test]
    fn power_off() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("192.168.1.1");
        robot.shutdown()?;
        Ok(())
    }

    #[test]
    fn enable() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("192.168.1.1");
        robot.enable()?;
        Ok(())
    }

    #[test]
    fn disable() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("192.168.1.1");
        robot.disable()?;
        Ok(())
    }
}
