# Planning for Learning Object Properties

This repository contains the official code of the Planning for Learning Object Properties (PLOP) algorithm, which has been presented at the 37th AAAI Conference on Artificial Intelligence (AAAI-2023 Main track), for details about the method please see the [paper](https://arxiv.org/pdf/2301.06054.pdf).


## Installation
The following instructions have been tested on Ubuntu 20.04.


1. Clone this repository
```
 git clone https://github.com/LamannaLeonardo/PLOP.git
```

2. Create a Python 3.9 virtual environment using conda or pip.
```
 conda create -n plop python=3.9
```

3. Activate the environment
```
 conda activate plop
```

4. Install [PyTorch](https://pytorch.org/get-started/locally/) (tested with version 1.10.0)

5. Install [AI2THOR](https://ai2thor.allenai.org/ithor/documentation) (tested with version 4.0.0) 
```
  pip install ai2thor
```

6. Install the following dependencies
```
pip install matplotlib scipy ipython pandas tqdm seaborn
```

7. Clone the yolov5 repository into the 'Utils' folder
```
cd PLOP/Utils && git clone https://github.com/ultralytics/yolov5.git 
```

8. Create the "pretrained_models" subdirectory into the "Utils" directory
```
mkdir pretrained_models && cd pretrained_models
```

9. Download the pretrained YoloV5 model available at this [link](https://drive.google.com/file/d/1eOJ3X6GG2_LAuzYgsHLUrVIvwclrDDta/view?usp=share_link), and move it into the directory "Utils/pretrained_models"


10. Download the [FastForward](https://fai.cs.uni-saarland.de/hoffmann/ff.html) planner in the directory "PLOP/PAL/Plan/PDDL/Planners/FF". 

 Create the directory "PLOP/PAL/Plan/PDDL/Planners/FF"
 ```
 cd ../../PAL/Plan/PDDL && mkdir -p Planners/FF && cd Planners/FF
 ```

 From the [offical FastForward site](https://fai.cs.uni-saarland.de/hoffmann/ff.html), download FF-v2.3.tgz (you can directly download it from this [link](https://fai.cs.uni-saarland.de/hoffmann/ff/FF-v2.3.tgz)) into the "PLOP/PAL/Plan/PDDL/Planners/FF" directory, 
 ```
 wget https://fai.cs.uni-saarland.de/hoffmann/ff/FF-v2.3.tgz
 ```

 Extract the archive ```tar -xf FF-v2.3.tgz```, go into the installation directory with ```cd FF-v2.3``` and compile FastForward with ```make```. 
 
 Finally move the "ff" executable in the parent directory through the command ```mv ff ../```, go to the parent directory ```cd ../``` and delete unnecessary files with ```rm -r FF-v2.3 & rm FF-v2.3.tgz```.


10. Check everything is correctly installed by running the "main.py" in the "PLOP" directory


## Execution

### Running PLOP
The PLOP algorithm can be run for learning to recognize the following object properties: dirty, open, filled, toggled (for further details about the tasks, please see the [paper](https://arxiv.org/pdf/2301.06054.pdf)). 
The learning task can be selected by setting the "TASK" flag in "Configuration.py".
To run PLOP on a specific task w/o ground truth object detections, there are two options:

e.g. to run PLOP on the task of learning to recognize the property 'dirty', set "TASK = TASK_LEARN_DIRTY" in "Configuration.py"


## Log and results
When you execute PLOP, a new directory with all logs and results is created in the "Results" folder. For instance, the logs and results are stored in the folder "Results/test_set_X_stepsY", where X is the task name set in "Configuration.py", and Y the number of steps (which can be set in "Configuration.py"). One subdirectory is created for each episode, which consists of a run in a single environment. Each episode subdirectory contains evaluation and log files relative to a single episode.


## Notes
1. The training sets collected online for each property and used for evaluation in the [paper](https://arxiv.org/pdf/2301.06054.pdf) can be downloaded at this [link](https://drive.google.com/file/d/1qJE1Xx2c_1a0tsDJfSiTkbgiTgP8lyPZ/view?usp=share_link)


## Citations
```
@inproceedings{Lamanna_AAAI_2023,
  title={Planning for Learning Object Properties},
  author={Lamanna, Leonardo and Serafini, Luciano and Mohamadreza, Faridghasmenia, and Saffiotti, Alessandro and Saetti, Alessandro and Gerevini, Alfonso Emilio and Traverso, Paolo},
  booktitle={Proceedings of the 37th AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](/License) file for details.
