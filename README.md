# Dreambooth-SD-Xformers-Win
A Windows compatible fork of ShivamShrirao/diffusers Dreambooth Xformers example with additional tools for ease of use.

## Conda installation
Open anaconda prompt and create env
```cmd
conda create -n diffusers python=3.8
conda activate diffusers
```

Install requirements
```cmd
conda install torchvision==0.13.1 -c pytorch -c conda-forge
pip install ./deps/diffusers-0.7.0.dev0-py3-none-any.whl
pip install ./deps/xformers-0.0.14.dev0-cp38-cp38-win_amd64.whl
pip install -r requirements.txt
```

---

Copy `deps/bitsandbytes-win-prebuilt/*` to `C:\Users\USER\.conda\envs\ENV_NAME\Lib\site-packages\bitsandbytes` so that the .dll are among the .so

Browse to `C:\Users\USER\.conda\envs\ENV_NAME\Lib\site-packages\bitsandbytes`

In `cextension.py` `~line 91`:

Replace 
```python
self.lib = ct.cdll.LoadLibrary(binary_path)
```
with 
```python
self.lib = ct.cdll.LoadLibrary(str(binary_path))
```


In `cuda_setup/main.py` `~line 119`:

Replace
```python
if not torch.cuda.is_available(): return 'libsbitsandbytes_cpu.so', None, None, None, None
```
with
```python
if torch.cuda.is_available(): return 'libbitsandbytes_cuda116.dll', None, None, None, None
if not torch.cuda.is_available(): return 'libsbitsandbytes_cpu.dll', None, None, None, None
```

---

Login to huggingface-cli and config accelerate
```cmd
huggingface-cli login
accelerate config
```

---

Train and convert using `dreambooth.bat`

---
## Workflow Example
Download any Stable Diffusion model to work off of (ie. [stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)), convert to diffuser with batch script, and place into `models` folder.
Place input images into `data/NAME/images`.
Generate class images and train with batch script. (optionally, when satisfied, convert model to SD/ckpt for use with [AUTOMATIC1111's webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and the such.


# Performance
## Important
Use table bellow to find best config for training. To adjust batch script config, edit `line 8-20` of `dreambooth.bat` to fit your needs. Backup `dreambooth.bat` stored in `deps` in case of damage.

---

Model with just [xformers](https://github.com/facebookresearch/xformers) memory efficient flash attention uses 15.79 GB VRAM with `--gradient_checkpointing` else 17.7 GB. Both have no loss in precision at all. gradient_checkpointing recalculates intermediate activations to save memory at cost of some speed.

Caching the outputs of VAE and Text Encoder and freeing them also helped in reducing memory.

Use the table below to choose the best flags based on your memory and speed requirements. Tested on Tesla T4 GPU.

| `fp16` | `train_batch_size` | `gradient_accumulation_steps` | `gradient_checkpointing` | `use_8bit_adam` | GB VRAM usage | Speed (it/s) |
| ---- | ------------------ | ----------------------------- | ----------------------- | --------------- | ---------- | ------------ |
| fp16 | 1                  | 1                             | TRUE                    | TRUE            | 9.92       | 0.93         |
| no   | 1                  | 1                             | TRUE                    | TRUE            | 10.08      | 0.42         |
| fp16 | 2                  | 1                             | TRUE                    | TRUE            | 10.4       | 0.66         |
| fp16 | 1                  | 1                             | FALSE                   | TRUE            | 11.17      | 1.14         |
| no   | 1                  | 1                             | FALSE                   | TRUE            | 11.17      | 0.49         |
| fp16 | 1                  | 2                             | TRUE                    | TRUE            | 11.56      | 1            |
| fp16 | 2                  | 1                             | FALSE                   | TRUE            | 13.67      | 0.82         |
| fp16 | 1                  | 2                             | FALSE                   | TRUE            | 13.7       | 0.83          |
| fp16 | 1                  | 1                             | TRUE                    | FALSE           | 15.79      | 0.77         |

# DreamBooth training example

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject.


```cmd
set MODEL_NAME="path-to-sd-model"
set INSTANCE_DIR="path-to-instance-images"
set OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py^
  --pretrained_model_name_or_path=%MODEL_NAME%^
  --instance_data_dir=%INSTANCE_DIR%^
  --output_dir=%OUTPUT_DIR%^
  --instance_prompt="a photo of sks dog"^
  --resolution=512^
  --train_batch_size=1^
  --gradient_accumulation_steps=1^
  --learning_rate=5e-6^
  --lr_scheduler="constant"^
  --lr_warmup_steps=0^
  --max_train_steps=400
```

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases.

```cmd
set MODEL_NAME="path-to-sd-model"
set INSTANCE_DIR="path-to-instance-images"
set CLASS_DIR="path-to-class-images"
set OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py^
  --pretrained_model_name_or_path=%MODEL_NAME%^
  --instance_data_dir=%INSTANCE_DIR%^
  --class_data_dir=%CLASS_DIR%^
  --output_dir=%OUTPUT_DIR%^
  --with_prior_preservation --prior_loss_weight=1.0^
  --instance_prompt="a photo of sks dog"^
  --class_prompt="a photo of dog"^
  --resolution=512^
  --train_batch_size=1^
  --gradient_accumulation_steps=1^
  --learning_rate=5e-6^
  --lr_scheduler="constant"^
  --lr_warmup_steps=0^
  --num_class_images=200^
  --max_train_steps=800
```


### Training on a 16GB GPU:

With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train dreambooth on a 16GB GPU.

```cmd
set MODEL_NAME="path-to-sd-model"
set INSTANCE_DIR="path-to-instance-images"
set CLASS_DIR="path-to-class-images"
set OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py^
  --pretrained_model_name_or_path=%MODEL_NAME%^
  --instance_data_dir=%INSTANCE_DIR%^
  --class_data_dir=%CLASS_DIR%^
  --output_dir=%OUTPUT_DIR%^
  --with_prior_preservation --prior_loss_weight=1.0^
  --instance_prompt="a photo of sks dog"^
  --class_prompt="a photo of dog"^
  --resolution=512^
  --train_batch_size=1^
  --gradient_accumulation_steps=2 --gradient_checkpointing^
  --use_8bit_adam^
  --learning_rate=5e-6^
  --lr_scheduler="constant"^
  --lr_warmup_steps=0^
  --num_class_images=200^
  --max_train_steps=800
```

### Training on a 8 GB GPU:

By using [DeepSpeed](https://www.deepspeed.ai/) it's possible to offload some
tensors from VRAM to either CPU or NVME allowing to train with less VRAM.

DeepSpeed needs to be enabled with `accelerate config`. During configuration
answer yes to "Do you want to use DeepSpeed?". With DeepSpeed stage 2, fp16
mixed precision and offloading both parameters and optimizer state to cpu it's
possible to train on under 8 GB VRAM with a drawback of requiring significantly
more RAM (about 25 GB). See [documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for more DeepSpeed configuration options.

Changing the default Adam optimizer to DeepSpeed's special version of Adam
`deepspeed.ops.adam.DeepSpeedCPUAdam` gives a substantial speedup but enabling
it requires CUDA toolchain with the same version as pytorch. 8-bit optimizer
does not seem to be compatible with DeepSpeed at the moment.

```cmd
set MODEL_NAME="path-to-sd-model"
set INSTANCE_DIR="path-to-instance-images"
set CLASS_DIR="path-to-class-images"
set OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py^
  --pretrained_model_name_or_path=%MODEL_NAME%^
  --instance_data_dir=%INSTANCE_DIR%^
  --class_data_dir=%CLASS_DIR%^
  --output_dir=%OUTPUT_DIR%^
  --with_prior_preservation --prior_loss_weight=1.0^
  --instance_prompt="a photo of sks dog"^
  --class_prompt="a photo of dog"^
  --resolution=512^
  --train_batch_size=1^
  --sample_batch_size=1^
  --gradient_accumulation_steps=1 --gradient_checkpointing^
  --learning_rate=5e-6^
  --lr_scheduler="constant"^
  --lr_warmup_steps=0^
  --num_class_images=200^
  --max_train_steps=800^
  --mixed_precision=fp16
```

### Fine-tune text encoder with the UNet.

The script also allows to fine-tune the `text_encoder` along with the `unet`. It's been observed experimentally that fine-tuning `text_encoder` gives much better results especially on faces. 
Pass the `--train_text_encoder` argument to the script to enable training `text_encoder`.

```cmd
set MODEL_NAME="path-to-sd-model"
set INSTANCE_DIR="path-to-instance-images"
set CLASS_DIR="path-to-class-images"
set OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py^
  --pretrained_model_name_or_path=%MODEL_NAME%^
  --train_text_encoder^
  --instance_data_dir=%INSTANCE_DIR%^
  --class_data_dir=%CLASS_DIR%^
  --output_dir=%OUTPUT_DIR%^
  --with_prior_preservation --prior_loss_weight=1.0^
  --instance_prompt="a photo of sks dog"^
  --class_prompt="a photo of dog"^
  --resolution=512^
  --train_batch_size=1^
  --use_8bit_adam
  --gradient_checkpointing^
  --learning_rate=2e-6^
  --lr_scheduler="constant"^
  --lr_warmup_steps=0^
  --num_class_images=200^
  --max_train_steps=800
```

# Credits
- Dreambooth Xformers - https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
- Bitsandbytes Prebuilt DLLs - https://github.com/DeXtmL/bitsandbytes-win-prebuilt
- Convert Diffusers to SD https://gist.github.com/jachiam/8a5c0b607e38fcc585168b90c686eb05