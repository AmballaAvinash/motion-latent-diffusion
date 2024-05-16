# CS674 Motion Synthesis project

**Instructions to setup the CS674 Motion Synthesis project. This repo has been built on top of the motion-latent-diffusion repo.**
<details>
<summary> <b>Install dependencies and dataset </b> </summary>

1. Conda environment
conda create python=3.9 --name mld
conda activate mld
Install the packages in requirements.txt and install PyTorch 2.2.2 If any package error, refer packages_ref.txt file

pip install -r requirements.txt
We test our code on Python 3.9.12 and PyTorch 2.2.2

2. Dependencies
Run the script to download dependencies materials:

bash prepare/download_smpl_model.sh
bash prepare/prepare_clip.sh
For Text to Motion Evaluation

bash prepare/download_t2m_evaluators.sh

3. Prepare the HumanML dataset
Please refer to HumanML3D for text-to-motion dataset setup.

You can download the Human ML data from our drive : https://drive.google.com/file/d/1bOmzxuH8xNk2XM4Onfde0tstDovgRowV/view?usp=sharing. However, note that due to the distribution policy of AMASS dataset, we are not allowed to distribute the data directly. We provided just for this project to reduce the effort in setting up the data and replicate the results. Unzip the data in a folder called datasets/humanml3d. Alternatively, follow the instructions given below to do the whole setup if you are interested.

    1. Download the dataset for this folder from amass data site. Navigate to this link of humanml3d repository (https://github.com/EricGuo5513/HumanML3D/tree/main)
    2. Run the scripts for raw_pose_processing.ipynb, motion_representation.ipynb, cal_mean_variance.ipynb.
    3. Follow the instructions of downloading datasets listed in the raw_pose_processing.ipynb. These datasets should be unzipped directly in a folder called amass_data/. For ex, the kitml dataset should be unzipped such that it follows this structure: amass_data/KIT/001/001.npy
    4. Make sure all the other files such as license.txt are removed. The datasets folders will have also have to be renamed to some specific names described in the raw_pose_processing.ipynb file.
    5. After running all 3 notebooks, you should end up with a folder called datasets/humanml3d which consists of npy files, text files, Mean.npy and Std.npy of the whole data. This should consists of motion representations collated from different data sources. It follows the SMPL skeleton structure of 22 joints.
    6. Make sure to run the verification cells in the above scripts so there aren't any errors on dataset setup.

</details>

<details>
<summary> <b>Repository Details</b> </summary>
1. Our Code Changes (and explaining the folder structure)
datasets

    - humanml3d: place the downloaded dataset here
    configs: contains the arguments

    - assets (modified this file to adjust the paths accordingly): path configs

    - config_vae_humanml3d.yaml (added this file): human ml 3d config file for VAE training

    - config_GAN_humanml3d.yaml (added this file): human ml 3d config file for GAN training
    prepare:

    - contains bash scripts to download the dependecies. Also make sure once you download edit the path accordingly in the configs/assest.yaml

mld

    - models: 

        - get_model.py: edited the line 6 in this file to handle modeltype GAN and WGAN 

        - model_type: 
            - base.py: base pytorch lighting module, mld.package
            - GAN.py (added this file): this files used in the train.py. Internally calls architectures/gan_arcitecture.py etc
            - WGAN.py (added this file):  this files used in the train.py. Internally calls architectures/wgan_arcitecture.py etc
            - WGANGP.py (added this file):  this files used in the train.py. Internally calls architectures/wgangp_basic.py etc


        - architectures: define architectures
            - gan_arcitecture.py (added this file): Has the simple GAN architecture with BCE loss
            - wgan_arcitecture.py (added this file):  Has the simple GAN architecture with Wassestein loss
            - gan_dense.py (added this file):  Has the Dense GAN architecture with BCE loss
            - wgan_dense.py (added this file): Has the Dense GAN architecture with Wassestein loss
            - mlp_gan.py (added this file): Has the MLP GAN architecture with BCE loss
            - wmlp_gan.py (added this file): Has the MLP GAN architecture with Wassestein loss
            - wgangp_basic.py (added this file): Has the simple GAN architecture with Wassestein loss and gradient penalty

        -losses: 
            - mld.py: added lines 50-55, 95-96, 136-139 to handle the stage "GAN"

    - train.py -> file in which training VAE or GAN happens.

    - trainer_bash.sh (added this file) -> To run the training in GPU

    - test.py -> testing and calculating the evaluation metrics

    - demo.py - loading the trained models and showing demo (text to motion)

    - render.py -> visulaize motions using blender

    - demo/example.txt: text input for testing

results:

    - GAN: results with GAN architectures

    - WGAN: results with WGAN architecture


</details>

<details>
<summary> <b>Important instructions to note before running scripts </b></summary>
We have set up config files depending on the model type you want to test (GAN, WGAN, WGANGP) and the architecture type.

The architecture type is set in model.arch_type in config files.

Given below is the valid architecture types you can test and demo:

stage  | architectures |
-------|-------------------|
GAN    | simple, dense, mlp |
WGAN   | simple, dense |
WGANGP | simple |

Use the appropriate config files based on the model stage as follows:

stage  | config file path |
-------|-------------------|
GAN    | ./configs/config_GAN_humanml3d.yaml |
WGAN   | ./configs/config_WGAN_humanml3d.yaml |
WGANGP | ./configs/config_WGANGP_humanml3d.yaml |

Make a checkpoints folder before running demo scripts so the models are downloaded and stored in the folder.

To make it easier for you to test the models, we have setup the demo script such that it will accept the model type and architecture type
from config files and automatically download the best model we have trained to the checkpoints folder.

</details>

<details>
<summary> <b>Train VAE </b> </summary>
Please first check the parameters in configs/config_vae_humanml3d.yaml, e.g. NAME

Then, run the following command:
```
python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
```
</details>
<details>
<summary> <b>Train GAN</b> </summary>
Ready to train GAN model?

Please update the parameters in configs/config_GAN_kitml.yaml, e.g. NAME, Update the PRETRAINED_VAE to the latest VAE ckpt model path in previous step
Use TRAIN.STAGE=GAN and model.model_type=GAN for GAN training, WGAN for Wassestein GAN training and WGANGP for Wasserstein GAN-GP respectively.
Please note you have to change both parameters for changes to work smoothly.

```
python -m train --cfg configs/config_GAN_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
```

change the config files depending on the model type (GAN, WGAN, WGANGP). Set the architecture in model.arch_type (simple, dense, mlp). Refer
to **Important Instructions** to get all valid architecture types for each model.

</details>

<details>
<summary> <b>Evaluate the model</b> </summary>
    Please first put the tained model checkpoint path to TEST.CHECKPOINT in configs/config_GAN_humanml3d.yaml.

    Then, run the following command:

    To test trained VAE: ```python -m test --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml```

    To test trained GAN: ```python -m test --cfg configs/config_GAN_humanml3d.yaml --cfg_assets configs/assets.yaml```

    change the config files depending on the model type (GAN, WGAN, WGANGP). Set the architecture in model.arch_type

</details>

<details>
<summary> <b>Demo</b> </summary>

**The following instructions are for getting visual outputs from the best models we have identified for each model type and architecture.**

**First set up blender in your system. Please follow the instructions given below to do the setup.**

1. Download and Install Blender (We used windows) - https://www.blender.org/download/releases/2-93/
2. Follow the Blender Installation procedure (from step 1-6 under visualization section) mentioned in the TEMOS-Rendering motions from the url - https://github.com/Mathux/TEMOS
3. Update the path environment variable to add the path of Blender(Blender.exe) by following the steps (for windows):

    (i) Select Start select Control Panel. double click System and select the Advanced tab.

    (ii) Click Environment Variables. In the section System Variables find the PATH environment variable and select it. Click Edit. If the PATH environment variable does not exist, click New.

    (iii) In the Edit System Variable (or New System Variable) window, specify the value of the PATH environment variable. Click OK. Close all remaining windows by clicking OK.

4. Execute the following command:
```YOUR_BLENDER_PYTHON_PATH/python -m pip install -r prepare/requirements_render.txt```
5. Download the checkpoint and deps folder from the following drive link (this is mentioned in the Quick start setup and download steps in the mld repo directly follow them if using Linux or Linux subsystem in windows): https://drive.google.com/drive/folders/1U93wvPsqaSzb5waZfGFVYc4tLCAOmB4C
unzip both the folders and move them into the repo directory
6. Open the config.py file in the directory motion-latent-diffusion-main\mld\transforms\joints2rots and update the SMPL_MODEL_DIR, GMM_MODEL_DIR, SMPL_MEAN_FILE and Part_Seg_DIR variables with respective file paths in your local systems.
7. Rename the render_mld.yaml file name into render.yaml in the configs folder.
8. Now execute the below command from the repo directory in command prompt
```blender --background --python render.py -- --cfg=./configs/render.yaml --dir=YOUR_NPY_FOLDER --mode=video --joint_type=HumanML3D```

In case the video generation fails but the frames are generated succefully, use blender to make video from the generated frames (reference video to do this: https://www.youtube.com/watch?v=jRsYkp3GoK0&ab_channel=BlenderInferno)

Make a folder called 'checkpoints' which will store all the necessary model checkpoints required to run this project and demo it.

Set the stage variable in TRAIN.STAGE inside the config files to set the GAN model type. To set the specific architecture (basic, dense, mlp) set it under
model.arch_type in the config files.

To run the demo using the trained GAN for the inputs demo/example.txt

Run demo using:  ```python demo.py --cfg configs/config_GAN_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt ```

The outputs:

npy file: the generated motions with the shape of (nframe, 22, 3) for HumanML. You can find these in the results section.
text file: the input text prompt

</details>

<details>
<summary><b>Outputs</b></summary>
You can find the npy converted to video here: https://drive.google.com/drive/folders/1Ik9CkRPsKm3_Gy8cMDZSJg1_K-qC8YDC?usp=drive_link
</details>

<details>
<summary><b>Contributions</b></summary>

- Avinash Amballa: Trained VAE, Setup the intiial code base and implmented the Simple GAN, Dense GAN with BCE loss. (gan_architecture.py, gan_dense.py, GAN.py)

- Vinitra Muralikrishnan: Implemented MLP GAN with BCE loss and Wassestein loss. Implemented Basic WGAN with Gradient Penality. (mlp_gan.py, wmlp_gan.py and wgangp_basic.py, WGANGP.py)

- Gayathri Akkinapalli: Implemented Simple GAN, Dense GAN with Wasserstein loss. Set up belder to render video from npy files. (wgan_architecture.py, wgan_dense.py, WGAN.py)
</details>