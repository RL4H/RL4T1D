import subprocess
import pytest

def test_run_rl_agent():
    """Test the RL agent script with PPO algorithm."""
    command = [
        "python", "run_RL_agent.py",
        "experiment.name=test1",
        "experiment.device=cpu",
        "agent=ppo",
        "agent.debug=True",
        "hydra/job_logging=disabled"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        assert "Experiment completed" in result.stdout  # Adjust based on expected output
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Command failed with error: {e.stderr}")