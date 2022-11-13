@echo off
title Dreambooth
echo Dreambooth
echo.
goto begin

:launch
accelerate launch train_dreambooth.py^
 --pretrained_model_name_or_path=%model_dir% --pretrained_vae_name_or_path=%model_dir%/vae^
 --instance_data_dir=data/%name%/images --class_data_dir data/%name%/class^
 --output_dir=%output_dir%^
 --with_prior_preservation --prior_loss_weight=1.0^
 --instance_prompt="%instance_prompt%" --class_prompt "%class_prompt%"^
 --num_class_images %class_count%^
 --seed=%random% --resolution 512 --train_batch_size 1^
 --train_text_encoder --mixed_precision="fp16" --use_8bit_adam^
 --gradient_accumulation_steps 1^ --learning_rate 1e-6^
 --lr_scheduler="constant" --lr_warmup_steps 0^
 --max_train_steps=%max_steps% --save_interval %save_interval% --save_sample_prompt="%instance_prompt%"^
 --n_save_sample %save_count% --save_infer_steps %save_steps%


exit /b



:begin
call conda activate diffusers


echo 1) New model
echo 2) Resume model
echo 3) Convert model
echo.
choice /c 123 /n /m "Choice: "

if errorlevel 3 goto convert
if errorlevel 2 goto resume
if errorlevel 1 goto new

:new
call :basic_prompts
set output_dir=data/%name%/model

goto train

:resume
call :basic_prompts
set /p step=Resume step: 

set model_dir=./data/%name%/model/%step%
set output_dir=data/%name%/model_%step%

goto train


:train
set /p class_count=Class count (%class_count%): 
echo.
set /p max_steps=Steps (%max_steps%): 
set /p save_interval=Save frequency (%save_interval%): 
set /p save_steps=Sample steps (%save_steps%): 
set /p save_count=Sample count (%save_count%): 

echo.

call :launch

echo.
pause
exit


:convert-diff
set /p in_path=Model path (eg. data/cutedog/model/1000): 
set /p out_path=Ckpt path (eg. data/cutedog/model/1000/model.ckpt): 
echo.

python convert_diff_to_sd.py --model_path "%in_path%" --checkpoint_path "%out_path%" --half

exit /b

:convert-sd
set /p in_path=Ckpt path (eg. data/sd-v1-4.ckpt): 
set /p out_path=Model path (eg. data/converted_sd-v1-4): 
echo.
choice /c yn /n /m "Extract EMA? "
echo.
if %errorlevel% equ 2 set ema=
if %errorlevel% equ 1 set ema=--extract_ema
python convert_sd_to_diff.py --checkpoint_path "%in_path%" --dump_path "%out_path%" %ema%

exit /b

:convert
echo.

echo 1) Convert Diff to SD
echo 2) Convert SD to Diff
echo.
choice /c 12 /n /m "Choice: "
echo.
if errorlevel 2 call :convert-sd
if errorlevel 1 call :convert-diff

echo.
echo Stored at %out_path%

echo.
pause
exit


:model_select
for /f "delims=" %%i in ('python -c "exec('''\nimport os\nfor i in range(len(next(os.walk('models'))[1])): print(i+1, end='')\n''')"') do set model_choices=%%i

echo.
echo.
echo Choose model
echo.
python -c "exec('''\nimport os\nfor i, x in enumerate(next(os.walk('models'))[1]): print(f'{i+1}) {x}')\n''')"
echo.
choice /c %model_choices% /n /m "Choice: "
set model_choice=%errorlevel%

for /f "delims=" %%i in ('python -c "exec('''\nimport os\nprint(next(os.walk('models'))[1][%model_choice%-1])\n''')"') do set model_name=%%i

exit /b


:set_prefs
set class_count=50
set model_dir=./models/%model_name%
set max_steps=1000
set save_interval=100
set save_steps=50
set save_count=2

exit /b

:name_prompt
echo.
set /p name=Name (eg. cutedog): 

exit /b

:basic_prompts
call :model_select
call :set_prefs
echo.
call :name_prompt
set /p instance_prompt=Prompt (eg. a photo of cutedog dog): 
set /p class_prompt=Class prompt (eg. a photo of dog): 

exit /b