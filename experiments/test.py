import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(cfg)
    if cfg.mlflow.track:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.experiment.name)
        experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
        run_name = cfg.experiment.run_name if cfg.experiment.run_name is not None else cfg.agent.agent

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
            mlflow.log_params(cfg)
            for i in range(0, 10):
                mlflow.log_metric("foo", 1+i)
                mlflow.log_metric("bar", 2+i)


if __name__ == "__main__":
    my_app()


# python test.py patient_id=2 experiment_folder=test2 +env=env_config +agent=ppo_config 'agent: debug_config'

# python test.py experiment.name=test experiment.device=cpu agent=ppo agent.debug=True hydra/job_logging=disabled
#  mlflow ui --backend-store-uri test/mlflow/
