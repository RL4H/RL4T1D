![header](img/header.png)
<h1>RL4T1D: Reinforcement Learning for Automating Treatment in Type 1 Diabetes.</h1>

![license](https://img.shields.io/badge/License-MIT-yellow.svg)
[![capsml](https://img.shields.io/badge/Web-CAPSML-red)](https://capsml.com/)
[![python_sup](https://img.shields.io/badge/python-3.10.12-black.svg?)](https://www.python.org/downloads/release/python-31012/)
[![CI](https://github.com/chirathyh/G2P2C/actions/workflows/python-app.yml/badge.svg)](https://github.com/chirathyh/G2P2C/actions/workflows/python-app.yml)
[![DOI](https://img.shields.io/badge/DOI-10.25911/CXAQ--3151-blue)](http://hdl.handle.net/1885/305591)

**IMPORTANT** This project is currently is in development: please visit the legacy codebase: [**G2P2C**](https://github.com/RL4H/G2P2C)

**RL4T1D** is a project to develop Reinforcement Learning (RL)-based Artificial Pancreas Systems (APS), with the aim to automate treatment in Type 1 Diabetes (T1D). For more detailed information please visit the legacy codebase: [**G2P2C**](https://github.com/RL4H/G2P2C)

**Background:** Type 1 Diabetes (T1D) is casued by the autoimmune destruction of the islet beta-cells and results in absolute insulin deficiency (cover image: Human islet of Langerhans created by Stable Diffusion). Hence, external administration of insulin is required to maintain glucose levels, which is cruicial as both low and high glucose levels are detrimental to health. This is usually done through an insulin pump attached to the body. An continuous glucose sensor is also attached to measure the glucose levels so that a control algorithm can estimate the appropriate insulin dose. In this project we design Reinforcement Learning (RL) based Artificial Pancreas Systems (APS) for the glucose control problem. The figure below shows the main components of an APS. 

<p align="center">
<img src="https://chirathyh.github.io/images/gif_aps.gif" width="70%" alt="APS">
</p>

**Objective/Task:** Maintaining glucose levels is a life-long optimisation problem, complicated due to the disturbances associated with daily events (meals, exercise, stress.. etc), delays present in glucose sensing and insulin action, partial observability, and safety constraints among others. A simulation of glucose regulation, using a RL-based strategy is shown below, where the optimal glucose range is shaded in green severe hypoglycemia / hyperglycemia ranges highlighted by the red dotted line. The blood glucose measurements are presented in the top, while the administered insulin by the RL agent is presented in the bottom. The disturbances related to meal events and the carbohydrate content of the meals are presented in red.

![Glucose Regulation](img/gif_glucose.gif)

Using the project
--

<h4>Installation</h4>

* Create and activate a Python3.10.12 virtual environment.
* Clone the repository: <code>git clone git@github.com:RL4H/RL4T1D.git</code>.<br>
* Go to the project folder (RL4T1D): <code>cd RL4T1D</code>
* Install the required Python libraries <code>requirements.txt</code>. 
* Create an environment file <code>.env</code> at the root of the project folder with <code>MAIN_PATH=path-to-this-project</code> (<code>echo "MAIN_PATH=$(pwd)">.env</code>).<br>

<h4>Prerequsites</h4>
* Install [Simglucosev0.2.2](https://github.com/jxx123/simglucose) which is an open source version of the UVA/PADOVA 2008 model approved by the FDA. <br>
* You can also install from the source (Recommended to install using <code>pip install -e .</code>, the simglucose 0.2.2 source code is available in the environments folder). The simulation environment and scenarios used in this project are extended from the original environment.

<h4>Quick Start</h4>

Running a **Proximal Policy Optimisation (PPO)** algorithm for glucose control. More information related to state-action space, reward formulations: [Paper](https://ieeexplore.ieee.org/abstract/document/9871054) .
```
cd experiments 
python run_RL_agent.py experiment.name=test33 experiment.device=cpu agent=ppo agent.debug=True hydra/job_logging=disabled
```
Running a **Soft Actor Critic (SAC)** algorithm for glucose control. 
```
cd experiments 
python run_RL_agent.py experiment.name=test33 experiment.device=cpu agent=sac agent.debug=True hydra/job_logging=disabled
```
Start mlflow
```
mlflow ui --backend-store-uri results/mlflow 
```

TODO: FIX: Broken due to env state space update. Running a clinical treatment strategy based on a clinical treatment strategy: **Basal-Bolus (BB)**. More info: [Paper](https://ieeexplore.ieee.org/abstract/document/9871054) .
```
cd experiments
python run_clinical_treatment.py --folder_id temp --carb_estimation_method real
```

<h4>Google Colab</h4> 
Check Docs/notebook. 

RoadMap and Notes
--
* Now you can use mlflow to organise: <experiments, experiment_runs (e.g., for hyperparams, new ideas)>. <br>
* We have legacy custom visualisation scripts <jupyter notebooks> used to debug experiments while running - integrate with new flow<br>
* Think/Complete logging. Duplicated with mlflow logs and legacy custom logging patterns <br>
  * experiment args
  * training logs (loss, gradients, etc)
  * worker logs (train, test, validate)
  * validation results.
* Write test cases <br>
* Add read docs like GluCoEnv. <br>
* Improve handling offpolicy vs onpolicy workers. <br>
* Issue running hydra from jupyter notebook. <br>
* build batch scripts to run for mutiple seeds, envs. <br>
* provide capability to run agents without using main script (integrate with other rl-libraries). <br>

### Citing
```
@misc{rl4t1d,
     author={RL4H Team},
     title={RL4T1D (2024)},
     year = {2024},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/RL4H/RL4T1D}},
   }
```
```
@article{hettiarachchi2023reinforcement,
  title={Reinforcement Learning-based Artificial Pancreas Systems to Automate Treatment in Type 1 Diabetes},
  author={Hettiarachchi, Chirath},
  year={2023},
  publisher={The Australian National University}
}
```
```
@article{hettiarachchi2024g2p2c,
  title={G2P2C—A modular reinforcement learning algorithm for glucose control by glucose prediction and planning in Type 1 Diabetes},
  author={Hettiarachchi, Chirath and Malagutti, Nicolo and Nolan, Christopher J and Suominen, Hanna and Daskalaki, Elena},
  journal={Biomedical Signal Processing and Control},
  volume={90},
  pages={105839},
  year={2024},
  publisher={Elsevier}
}
```

Contact
--
Chirath Hettiarachchi - chirath.hettiarachchi@anu.edu.au\
School of Computing, College of Engineering & Computer Science,\
Australian National University. 

