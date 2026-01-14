use std::{thread::sleep, time::Duration};

use anyhow::{Ok, Result};
use libjaka::JakaMini2;
use robot_behavior::behavior::*;

fn main() -> Result<()> {
    let mut robot = JakaMini2::new("192.168.1.1");

    robot.move_joint(&[
        -1.19742476853498,
        0.018282708915888905,
        2.0349804153426794,
        2.9468296676164088e-06,
        1.0884027170493922,
        0.17453292519943334,
    ])?;

    robot.move_traj_from_file("robot_draw\\img\\step09_stable_traj_rust.json")?;

    sleep(Duration::from_secs(10000));

    Ok(())
}

#[cfg(test)]
mod test {
    use libjaka::JakaMini2;
    use robot_behavior::behavior::*;

    #[test]
    fn init() {
        let mut robot = JakaMini2::new("192.168.1.1");
        robot.init();
    }

    #[test]
    fn enable() {
        let mut robot = JakaMini2::new("192.168.1.1");
        robot.enable();
    }
}
