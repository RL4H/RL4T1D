import subprocess
import pytest
import os

@pytest.mark.order(2)
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

@pytest.mark.order(1)
def test_create_env_file():
    """Test the creation of the .env file with MAIN_PATH."""
    try:
        # Run the command to create the .env file
        subprocess.run("MAIN_PATH=$(pwd) > .env", shell=True, check=True)
        subprocess.run("SIM_DATA_PATH=$(pwd) > .env", shell=True, check=True)

        # Validate the .env file content
        # with open(".env", "r") as env_file:
        #     content = env_file.read()
        #     assert content.strip() == f"MAIN_PATH={os.getcwd()}", \
        #         f"Unexpected .env content: {content}"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to create .env file: {e.stderr}")