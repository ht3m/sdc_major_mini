use std::{thread::sleep, time::Duration};

use anyhow::{Ok, Result};
use libjaka::JakaMini2;
use robot_behavior::behavior::*;

fn main() -> Result<()> {
    let mut robot = JakaMini2::new("10.5.5.100");

    robot.move_joint(&[
        0.43642145501756757,
        0.043303908409149966,
        -1.9405468867228346,
        -1.4748536326574774e-05,
        -1.2443031821789015,
        0.4364556645834642,
    ])?;

    robot.move_traj_from_file("robot_draw\\img\\step08_optimized_trajectory_rust.json")?;

    sleep(Duration::from_secs(100));

    Ok(())
}

#[cfg(test)]
mod test {
    use libjaka::JakaMini2;
    use robot_behavior::behavior::*;

    #[test]
    fn init() {
        let mut robot = JakaMini2::new("10.5.5.100");
        robot.init();
    }

    #[test]
    fn enable() {
        let mut robot = JakaMini2::new("10.5.5.100");
        robot.enable();
    }
}
