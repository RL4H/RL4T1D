import subprocess
import pytest

def test_run_rl_agent():
    """Test the RL agent script with PPO algorithm."""
    command = [
        "python", "experiments/run_RL_agent.py",
        "experiment.name=test1",
        "experiment.device=cpu",
        "agent=ppo",
        "agent.debug=True",
        "hydra/job_logging=disabled"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout

        # Check for the presence of the summary table and success message
        assert "===> Starting Validation Trials ...." in output, "Validation trials not started."
        assert "Algorithm Training/Validation Completed Successfully." in output, "Algorithm did not complete successfully."
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Command failed with error: {e.stderr}")