# RD-tools

this is a modular app that can handle some repetitive tasks in the RD process

## Functions

### DICOM Processing
Module Function: Converts medical imaging DICOM files into images or videos, saving them in folders mirroring the original DICOMDIR hierarchy.

#### UI Workflow
Select the working directory, which should be the grandparent directory of the DICOMDIR file.
Tick the target subdirectories in the checkbox list.
Find the DICOM Processing task card in the File window.
Click the "Execute Task" button on that card.
Monitor the dedicated log in the same card. A "SUCCESS! log file saved." message indicates completion.

#### CLI Workflow
Run the task module through Python's module entrypoint:

```bash
python -m modules.dicom_to_imgs
```

This CLI path uses the same user settings file as the GUI and follows the same parameter assembly logic.

Customization: If your DICOM file structure differs from the assumed hierarchy, rewrite the main function in dicom_to_imgs.py by leveraging the DicomToImage class and base functions in file_basic.py to tailor processing for your specific structure.

### ECG Signal Processing
Module Function: Processes single-lead ECG data (saved as CSV files) from an electrode, generating the following outputs:

Raw data: Time-domain and frequency-domain plots.
Primary filtered data: Time-domain, frequency-domain, and comparison plots.
Advanced processing:
- R-wave peak detection.
- PQRST waveform detection.
- Heart rate calculation.

Steps:
Set the sampling rate (default: 1000 Hz) to match your data.
Select the parent directory of the ECG data folder as the working directory.
Tick the target subdirectories in the checkbox list and run the ECG task card.


### Bilibili Video Export
Function: Batch exports cached Bilibili app videos into MP4 format (similar to yt-dlp but optimized for cached files).
Reference: BilibiliCacheVideoMergePython.


### Caption Generation
Core: Modified implementation based on VideoCaptioner.
Features:

Supports local configuration of whisper models (e.g., faster-whisper for PotPlayer).
Requires manual setup of dependencies:
whisper-cpp (Linux/Mac recommended).
faster-whisper (Windows recommended).
Limitations:
Translation functionality is removed.


### RGB Channel Decomposition/Synthesis
Functions:
- SplitColors: Separates RGB channels of images (useful for fluorescence-labeled images).
- MergeColors: Merges specified channels (e.g., R+G) from image pairs into composite images.


### TwistImgs
Function: Applies perspective distortion to images, creating quadrilateral visual effects (useful for custom poster designs or scene mockups).
Note: Specialized for niche use cases.


### more is comming


## How to install

1. Install [Python](https://www.python.org/downloads/) (make sure to add it to the environment variables).
2. (Optional) Create and activate a Python virtual environment.
3. install python third party libs by running the following command in terminal (select1)

```bash
python install.py
```

4. Install [ffmpeg](https://www.ffmpeg.org/download.html) (make sure to add it to the environment variables).
5. install [git](https://git-scm.com/downloads) (make sure to add it to the environment variables).
6. open terminal in a folder you like, and running the following command in terminal

```bash
git clone https://github.com/waaall/RD-tools.git
```

## How to update

open terminal in the `RD-tools` folder, and running the following command in terminal

```bash
git pull
```

NOTICE: if you change the code, maybe have errors when you running `git pull` command. You need to learn how to fix the git conflict.Since you have modified the code, I assume you possess many fundamental skills, such as being proficient in Git and Python; otherwise, you can reinstall.

## How to Use

 start this app by running the following command in terminal.

```bash
python main.py
```

Run a single task from terminal with:

```bash
python -m modules.gen_subtitles
```

GUI and CLI both read the same user settings from `~/Develop/RD-tools-configs/settings.json`.

For more details, please refer to the Help window within the app.
